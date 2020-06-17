"""Train an INN classifier on CIFAR-10.

Example
-------
python train_inn_classifier.py \
    --coupling=affine \
    --permutation=conv \
    --zero_init \
    --use_prior \
    --use_actnorm \
    --no_inverse_svd \
    --fd_coeff=1e-4 \
    --fd_inv_coeff=1e-4 \
    --regularize_every=10 \
    --mem_saving
    --save_dir=saves/affine_conv
"""
import os
import sys
import ipdb
import time
import random
import datetime
import argparse
import pickle as pkl
from tqdm import tqdm

import numpy as np

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms
from torchvision import utils as torchvision_utils

# Local imports
import utils
import plot_utils
import data_loaders
from csv_logger import CSVLogger, plot_csv
from invertible_layers2_mem import Glow_


# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# Training
parser.add_argument('--n_epochs', type=int, default=200,
                    help='Number of epochs to train')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='Dataset')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--coupling', type=str, default='additive', choices=['additive', 'affine'],
                    help='Type of coupling (additive or affine)')
parser.add_argument('--permutation', type=str, default='shuffle', choices=['shuffle', 'conv'],
                    help='Type of permutation (shuffle or conv)')
parser.add_argument('--n_levels', type=int, default=3,
                    help='Number of levels in the model')
parser.add_argument('--depth', type=int, default=16,
                    help='Depth of each block')
parser.add_argument('--hidden_channels', type=int, default=128,
                    help='Number of hidden channels in NN')
parser.add_argument('--use_actnorm', action='store_true', default=False,
                    help='Whether to use actnorm BETWEEN BLOCKS in the flow.')
parser.add_argument('--use_actnorm_in_blocks', action='store_true', default=False,
                    help='Whether to use actnorm INSIDE BLOCKS in the flow.')
parser.add_argument('--an_stable_eps', type=float, default=0,
                    help='ActNorm stabilizing epsilon')
parser.add_argument('--affine_scale_fn', type=str, default='sigmoid', choices=['sigmoid', 'exp'],
                    help='')
parser.add_argument('--n_bits_x', type=int, default=8.,
                    help='')
parser.add_argument('--inj_pad', type=int, default=0,
                    help='Injective padding')
parser.add_argument('--learntop', action='store_true', default=False,
                    help='')
parser.add_argument('--affine_eps', type=float, default=0.0,
                    help='Epsilon added to the scale in the affine coupling layers: z2 = z2 / (scale + self.affine_eps)')

# Optimization hyperparameters
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for the flow')
parser.add_argument('--lr_decay', type=float, default=0.1,
                    help='Factor by which to multiply the learning rate whenever lr decays')
parser.add_argument('--wdecay', type=float, default=0,
                    help='Weight decay for the flow optimization')

parser.add_argument('--use_prior', action='store_true', default=False,
                    help='Whether to add the prior term to the objective')
parser.add_argument('--nf_coeff', type=float, default=-1,
                    help='Strength of the log-det regularizer (-1 means it is not used)')
parser.add_argument('--fd_coeff', type=float, default=0.0,
                    help='Choose the strength of the forward finite-differences regularizer')
parser.add_argument('--fd_inv_coeff', type=float, default=0.0,
                    help='Choose the strength of the inverse finite-differences regularizer')
parser.add_argument('--regularize_every', type=int, default=1,
                    help='How often to add the regularization terms to the objective (every N iterations)')
parser.add_argument('--mem_saving', action='store_true', default=False,
                    help='Use memory-saving training')

parser.add_argument('--zs_dim', type=int, default=3072,
                    help='The number of dimensions to use for z_s')
parser.add_argument('--relu_inplace', action='store_true', default=False,
                    help='Choose whether to apply ReLU in-place or not, inside the blocks.')
parser.add_argument('--zero_init', action='store_true', default=False,
                    help='Use this flag to use zero init for the last conv layer in each block')

# Logging
parser.add_argument('--eval_every', type=int, default=2,
                    help='How often to evaluate accuracy on the full training and test sets')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save the model every N iterations')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Save a model checkpoint every N epochs')
parser.add_argument('--log_every', type=int, default=400,
                    help='Log training statistics every N iterations')
parser.add_argument('--plot_csv', action='store_true', default=False,
                    help='Whether to plot all the iteration CSV files')
parser.add_argument('--svd_every', type=int, default=2000,
                    help='Compute the SVD of the Jacobian every N iterations')
parser.add_argument('--save_dir', type=str, default='saves',
                    help='Directory for log / saving')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
parser.add_argument('--no_svd', action='store_true', default=False,
                    help='Save time by omitting SVD computation')
parser.add_argument('--no_inverse_svd', action='store_true', default=False,
                    help='Save time by omitting SVD computation of the Jacobian of the inverse mapping')
args = parser.parse_args()

args.n_bins = 2 ** args.n_bits_x  # 2^8 = 256

use_device = 'cuda:0'

# Set random seed for reproducibility
if args.seed is not None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
exp_name = '{}-{}-d:{}-h:{}-c:{}-p:{}-act:{}-fd:{}-jfd:{}-nf:{}-r:{}-mem:{}-s:{}'.format(
            timestamp, args.dataset, args.depth, args.hidden_channels, args.coupling, args.permutation,
            int(args.use_actnorm), args.fd_coeff, args.fd_inv_coeff, args.nf_coeff,
            args.regularize_every, int(args.mem_saving), args.seed)

save_dir = os.path.join('saves', args.save_dir, exp_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'samples'))
    os.makedirs(os.path.join(save_dir, 'samples_three_grid'))  # To save small 3x3 grids
    os.makedirs(os.path.join(save_dir, 'samples_fixed'))
    os.makedirs(os.path.join(save_dir, 'samples_fixed_three_grid'))  # To save small 3x3 grids
    os.makedirs(os.path.join(save_dir, 'samples_np'))  # To numpy full sample array
    os.makedirs(os.path.join(save_dir, 'samples_fixed_np'))  # To numpy full sample array for fixed minibatch
    os.makedirs(os.path.join(save_dir, 'norms'))

# Save command-line arguments
with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)

# Set up logging
iteration_fieldnames = ['global_iteration', 'time_elapsed', 'xentropy_loss', 'loss', 'train_acc',
                        'w_norm', 'w_min', 'w_max', 'w_mean', 'w_std',
                        'g_norm', 'g_min', 'g_max', 'g_mean', 'g_std',
                        'z_norm', 'z_min', 'z_max', 'z_mean', 'z_std']

every_N_fieldnames = ['global_iteration', 'time_elapsed', 'xentropy_loss', 'loss', 'train_acc',
                      'min_recons_error', 'max_recons_error', 'mean_recons_error', 'std_recons_error',
                      'w_norm', 'w_min', 'w_max', 'w_mean', 'w_std',
                      'g_norm', 'g_min', 'g_max', 'g_mean', 'g_std',
                      'z_norm', 'z_min', 'z_max', 'z_mean', 'z_std' ]

svd_fieldnames = ['global_iteration', 'mean_recons_error',
                  'condition_num', 'max_sv', 'min_sv',
                  'inverse_condition_num', 'inverse_max_sv', 'inverse_min_sv']

epoch_fieldnames = ['epoch', 'time_elapsed', 'train_loss', 'test_loss', 'train_acc', 'test_acc',
                    'mean_train_recons_error', 'std_train_recons_error',
                    'mean_test_recons_error', 'std_test_recons_error']

iteration_logger = CSVLogger(fieldnames=iteration_fieldnames,
                             filename=os.path.join(save_dir, 'iteration_log.csv'))
every_N_logger = CSVLogger(fieldnames=every_N_fieldnames,
                           filename=os.path.join(save_dir, 'every_N_log.csv'))
svd_logger = CSVLogger(fieldnames=svd_fieldnames,
                       filename=os.path.join(save_dir, 'svd_log.csv'))
epoch_logger = CSVLogger(fieldnames=epoch_fieldnames,
                         filename=os.path.join(save_dir, 'epoch_log.csv'))


# Data loading
# ------------
cifar_mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
cifar_std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=cifar_mean, std=cifar_std)])

tf_test = transforms.Compose([transforms.Resize(32),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=cifar_mean, std=cifar_std)])

imsize = 32
channels = 3
num_classes = 10

train_loader, test_loader, init_loader = data_loaders.load_cifar10(args.batch_size, tf, tf_test)


# Build models
# ------------
model = Glow_((args.batch_size, channels, imsize, imsize), args)
model = nn.DataParallel(model.cuda())

print('Number of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))

projection = nn.Sequential(nn.BatchNorm1d(args.zs_dim),
                           nn.ReLU(inplace=True),
                           nn.Linear(args.zs_dim, num_classes))
projection = projection.cuda()

params = list(model.parameters()) + list(projection.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

# Initialize actnorm
with torch.no_grad():
  model.eval()
  for (img, _) in init_loader:
      img = img.to(use_device)
      objective = torch.zeros_like(img[:, 0, 0, 0])
      _ = model(img, objective)
      break


def save_models(name='checkpoint'):
    torch.save(model, os.path.join(save_dir, 'model_{}.pt'.format(name)))  # Save flow model
    torch.save(projection, os.path.join(save_dir, 'projection_{}.pt'.format(name)))  # Save projection


def run_flow(img):
    objective = torch.zeros_like(img[:, 0, 0, 0])
    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    return model(img, objective)


def evaluate(dataloader, epoch):
    losses = []
    num_correct = 0.
    num_total = 0.
    all_recons_errors = []  # Accumulates all reconstruction errors over the test set

    model.eval()
    projection.eval()

    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(dataloader)):
            img, label = img.to(use_device), label.to(use_device)

            z, objective = run_flow(img)
            zs = z.view(z.size(0), -1)[:, :args.zs_dim]

            # Compute reconstruction error
            reconstructed = model.module.sample(z)
            recons_diff = img - reconstructed
            recons_errors = torch.norm(recons_diff.view(recons_diff.size(0), -1), dim=1)
            all_recons_errors.append(recons_errors)

            logits = projection(zs)
            classification_loss = F.cross_entropy(logits, label)
            losses.append(classification_loss.item())

            num_correct += torch.eq(logits.argmax(dim=1), label).sum().item()
            num_total += label.size(0)

        acc = num_correct / float(num_total)
        print('Test acc: {:6.4f}'.format(acc))

        all_recons_errors = torch.cat(all_recons_errors)
        mean_test_recons_error = torch.mean(all_recons_errors).item()
        std_test_recons_error = torch.std(all_recons_errors).item()

    mean_loss = np.mean(losses)
    return mean_loss, acc, mean_test_recons_error, std_test_recons_error


start_time = time.time()
global_iteration = 0
i = 0  # Update iter
largest_condition_num = 0  # Keeps track of the largest cond num so far

# Manually get a minibatch to use for visualization
fixed_img = torch.stack([test_loader.dataset[idx][0] for idx in range(192,320)])
fixed_img = fixed_img.to(use_device)

save_models('initialization')
decay_epochs = [60, 120, 160]

# Training loop
# ------------------------------------------------------------------------------
try:
    for epoch in range(args.n_epochs):
        if epoch in decay_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay  # Divide the lr by a factor of 10 at epochs 60, 120, 160
            print('='*80)
            print('Updated learning rate to: {:6.4e}'.format(optimizer.param_groups[0]['lr']))
            print('='*80)

        num_total = 0.0
        num_correct = 0.0

        progress_bar = tqdm(train_loader)
        for _, (img, label) in enumerate(progress_bar):
            i += 1
            progress_bar.set_description('Epoch ' + str(epoch))

            model.train()
            projection.train()

            img, label = img.to(use_device), label.to(use_device)

            z, objective = run_flow(img)
            zs = z.view(z.size(0), -1)[:, :args.zs_dim]

            loss = 0
            logits = projection(zs)
            classification_loss = F.cross_entropy(logits, label)
            loss += classification_loss
            objective_mean = objective.mean()

            if args.nf_coeff > 0 and (global_iteration % args.regularize_every == 0):
                loss += -args.nf_coeff * objective_mean

            # For forward finite-differences regularization
            # -------------------------------------------
            if (args.fd_coeff > 0) and (global_iteration % args.regularize_every == 0):
                d = torch.randn(img.shape, device=img.device)
                d = (d.view(d.shape[0], -1) / torch.norm(d.view(d.shape[0], -1), dim=0)).view(d.shape)  # normalize to unit vector
                epsilon = 0.1
                z_noise, _ = run_flow(img + epsilon * d)
                J_forward_penalty = args.fd_coeff * torch.norm(z - z_noise)**2 / epsilon
                loss += J_forward_penalty
            else:
                J_forward_penalty = 0.0
            # -------------------------------------------

            # For inverse finite-differences regularization
            # -------------------------------------------
            if (args.fd_inv_coeff > 0) and (global_iteration % args.regularize_every == 0):
                d_inv = torch.randn(z.shape, device=z.device)
                d_inv = (d_inv.view(d_inv.shape[0], -1) / torch.norm(d_inv.view(d_inv.shape[0], -1), dim=0)).view(d_inv.shape)
                epsilon = 0.1
                z_noise_inv = z + epsilon * d_inv
                x_inv_noise = model.module.sample(z_noise_inv, no_grad=True)
                x_inv_noise.requires_grad = True
                J_inverse_penalty = args.fd_inv_coeff * torch.norm(img - x_inv_noise)**2 / epsilon
                loss += J_inverse_penalty
            # -------------------------------------------

            if args.mem_saving:
                if ((args.fd_inv_coeff > 0) or (args.fd_coeff > 0)) and (global_iteration % args.regularize_every == 0):
                    loss_forward_penalty = classification_loss + J_forward_penalty
                    z_grad = torch.autograd.grad(loss_forward_penalty, z, retain_graph=True)[0]
                    model_gradient_original = model.module.gradient(z, z_grad)
                    model_gradient = [g for g in model_gradient_original if g is not None]

                    # Gradient computation for forward JR
                    # -----------------------------------------------------------------
                    if args.fd_coeff > 0:
                        z_grad_noise = torch.autograd.grad(loss_forward_penalty, z_noise, retain_graph=True)[0]
                        model_gradient_noise = model.module.gradient(z_noise, z_grad_noise)
                        model_gradient_noise = [g for g in model_gradient_noise if g is not None]
                        model_gradient = [mg_original + mg_noise for (mg_original, mg_noise) in zip(model_gradient, model_gradient_noise)]
                        # -----------------------------------------------------------------

                    if args.fd_inv_coeff > 0:
                        # Gradient computation for inverse JR
                        # -----------------------------------------------------------------
                        x_inv_grad_noise = torch.autograd.grad(J_inverse_penalty, x_inv_noise, retain_graph=True)[0]
                        model_gradient_x_inv_noise, recon = model.module.gradient_forward(x_inv_noise,x_inv_grad_noise, z, d_inv, epsilon)
                        recon_error = (torch.norm(recon - img) / img.shape[0]).item()
                        print('Recon error after gradient: ' + str(recon_error))
                        model_gradient_x_inv_noise = [g for g in model_gradient_x_inv_noise if g is not None]
                        model_gradient = [mg + mg_inv for (mg, mg_inv) in zip(model_gradient, model_gradient_x_inv_noise)]
                        # -----------------------------------------------------------------
                else:
                    z_grad = torch.autograd.grad(classification_loss, z, retain_graph=True)[0]
                    model_gradient = model.module.gradient(z, z_grad)
                    model_gradient = [g for g in model_gradient if g is not None]

                projection_gradient = torch.autograd.grad(loss, projection.parameters(), retain_graph=True)
                for (param, g) in zip(projection.parameters(), projection_gradient):
                    param.grad = g.detach()

                for (param, g) in zip(model.parameters(), model_gradient):
                    param.grad = g.detach()

                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_iteration % args.save_every == 0:
                save_models('save_every_ep_{}_iter_{}'.format(epoch, global_iteration))

            num_total += logits.size(0)
            num_correct += (torch.argmax(logits, dim=1) == label).sum().float()

            # Measure weight norm and gradient norm
            # -------------------------------------
            flat_weights = torch.cat([p.view(-1) for p in model.parameters()])
            weight_norm = torch.norm(flat_weights)
            weight_norm_list = [torch.norm(p.view(-1)).item() for p in model.parameters()]

            flat_gradients = torch.cat([p.grad.view(-1) for p in model.parameters()])
            gradient_norm = torch.norm(flat_gradients)
            gradient_norm_list = [torch.norm(p.grad.view(-1)).item() for p in model.parameters()]

            z_norm = torch.norm(z.view(z.size(0), -1))
            # -------------------------------------

            stats_dict = { 'global_iteration': global_iteration,
                           'time_elapsed': time.time() - start_time,
                           'xentropy_loss': torch.mean(classification_loss).item(),
                           'loss': torch.mean(loss).item(),

                           'w_norm': weight_norm.item(),
                           'w_min': flat_weights.min().item(),
                           'w_max': flat_weights.max().item(),
                           'w_mean': flat_weights.mean().item(),
                           'w_std': flat_weights.std().item(),

                           'g_norm': gradient_norm.item(),
                           'g_min': flat_gradients.min().item(),
                           'g_max': flat_gradients.max().item(),
                           'g_mean': flat_gradients.mean().item(),
                           'g_std': flat_gradients.std().item(),

                           'z_norm': z_norm.item(),
                           'z_min': z.view(z.size(0), -1).min().item(),
                           'z_max': z.view(z.size(0), -1).max().item(),
                           'z_mean': z.view(z.size(0), -1).mean().item(),
                           'z_std': z.view(z.size(0), -1).std().item(),

                           'train_acc': (num_correct / num_total).item() }


            iteration_logger.writerow(stats_dict)
            progress_bar.set_postfix(stats_dict)


            if (i + 1) % args.log_every == 0:

                # Save the lists of weight and gradient norms per layer
                with open(os.path.join(save_dir, 'norms', 'w_norm_{}_{}.pkl'.format(epoch, global_iteration)), 'wb') as f:
                    pkl.dump(weight_norm_list, f)

                with open(os.path.join(save_dir, 'norms', 'g_norm_{}_{}.pkl'.format(epoch, global_iteration)), 'wb') as f:
                    pkl.dump(gradient_norm_list, f)

                model.eval()
                projection.eval()

                with torch.no_grad():
                    z, _ = run_flow(img)
                    z_shape = z.size()

                    for (original_img, sample_dir) in [(img, 'samples'), (fixed_img, 'samples_fixed')]:
                        z, _ = run_flow(original_img)
                        # To visualize reconstruction error
                        reconstructed = model.module.sample(z)

                        recons_diff = torch.abs(original_img - reconstructed)
                        recons_errors = torch.norm(recons_diff.view(recons_diff.size(0), -1), dim=1)

                        recons_err_viz = recons_diff
                        for channel in range(3):
                            recons_err_viz[:,channel].sub_(cifar_mean[channel]).div_(cifar_std[channel])

                        full_sample = torch.cat((original_img, reconstructed, recons_err_viz), dim=0)
                        sample = torch.cat((original_img[:10], reconstructed[:10], recons_err_viz[:10]), dim=0)
                        three_sample = torch.cat((original_img[:3], reconstructed[:3], recons_err_viz[:3]), dim=0)

                        for channel in range(3):
                            sample[:,channel].mul_(cifar_std[channel]).add_(cifar_mean[channel])
                            three_sample[:,channel].mul_(cifar_std[channel]).add_(cifar_mean[channel])
                        sample.clamp_(min=0, max=1)
                        three_sample.clamp_(min=0, max=1)

                        if sample_dir == 'samples_fixed':  # only save the fixed mini-batch samples to save disk space
                            np.save(os.path.join(save_dir, '{}_np'.format(sample_dir), 'sample_{}_{}.npy'.format(epoch, global_iteration)), full_sample.detach().cpu().numpy())

                        grid = torchvision_utils.make_grid(sample, nrow=10, padding=2, pad_value=1.0)
                        three_grid = torchvision_utils.make_grid(three_sample, nrow=3, padding=2, pad_value=1.0)
                        torchvision_utils.save_image(grid, os.path.join(save_dir, '{}'.format(sample_dir), 'sample_{}_{}.png'.format(epoch, global_iteration)))
                        torchvision_utils.save_image(three_grid, os.path.join(save_dir, '{}_three_grid'.format(sample_dir), 'sample_{}_{}.png'.format(epoch, global_iteration)))

                stats_dict['min_recons_error'] = recons_errors.min().item()
                stats_dict['max_recons_error'] = recons_errors.max().item()
                stats_dict['mean_recons_error'] = recons_errors.mean().item()
                stats_dict['std_recons_error'] = recons_errors.std().item()
                every_N_logger.writerow(stats_dict)


                if not args.no_svd and (i + 1) % args.svd_every == 0:
                    try:
                        svd_dict = {}
                        svd_dict['global_iteration'] = global_iteration
                        svd_dict['mean_recons_error'] = recons_errors.mean().item()

                        U, D, V = utils.computeSVDjacobian(fixed_img, model, inverse=False)

                        svd_dict['condition_num'] = float(D.max() / D.min())
                        svd_dict['max_sv'] = float(D.max())  # Because D.max() is a numpy.float32 value
                        svd_dict['min_sv'] = float(D.min())

                        if not args.no_inverse_svd:  # Save computation by omitting the inverse SVD computation
                            U_i, D_i, V_i = utils.computeSVDjacobian(fixed_img, model, inverse=True)

                            svd_dict['inverse_condition_num'] = float(D_i.max() / D_i.min())
                            svd_dict['inverse_max_sv'] = float(D_i.max())  # Because D.max() is a numpy.float32 value
                            svd_dict['inverse_min_sv'] = float(D_i.min())

                        if svd_dict['condition_num'] > largest_condition_num:
                            largest_condition_num = svd_dict['condition_num']
                            save_models('iter_{}_cond_{}'.format(global_iteration, svd_dict['condition_num']))

                        svd_logger.writerow(svd_dict)
                        plot_utils.plot_stability_stats(save_dir)
                        plot_utils.plot_individual_figures(save_dir, 'svd_log.csv')  # Makes separate PDFs for each logged measure
                    except:
                        print('Something went wrong when computing the SVD...')


                try:
                    plot_utils.plot_individual_figures(save_dir, 'every_N_log.csv')  # Makes separate PDFs for each logged measure
                except:
                    pass

                if args.plot_csv:
                    try:
                        plot_csv(iteration_logger.filename, key_name='global_iteration', yscale='log')
                    except:
                        pass

                model.train()
                projection.train()

            if torch.isnan(loss):
                print('='*80)
                print('Loss is NaN. Quitting...')
                print('='*80)
                sys.exit(1)

            global_iteration += 1

        if (epoch+1) % args.eval_every == 0:
            train_loss, train_acc, mean_train_recons_error, std_train_recons_error = evaluate(train_loader, epoch)
            test_loss, test_acc, mean_test_recons_error, std_test_recons_error = evaluate(test_loader, epoch)

            print('End of epoch {} | train acc: {:6.4f} | test acc: {:6.4f}'.format(epoch, train_acc, test_acc))

            epoch_logger.writerow({'epoch': epoch,
                                   'time_elapsed': time.time() - start_time,
                                   'train_loss': train_loss,
                                   'test_loss': test_loss,
                                   'train_acc': train_acc,
                                   'test_acc': test_acc,
                                   'mean_train_recons_error': mean_train_recons_error,
                                   'std_train_recons_error': std_train_recons_error,
                                   'mean_test_recons_error': mean_test_recons_error,
                                   'std_test_recons_error': std_test_recons_error
                                  })

            plot_csv(epoch_logger.filename, key_name='epoch')

        # Save generator/flow and discriminator at each epoch
        # if epoch % args.checkpoint_every == 0:
        #     save_models('ep_{}_iter_{}'.format(epoch, global_iteration))

except KeyboardInterrupt:
    print('='*80)
    print('Exiting training early...')
    print('='*80)
