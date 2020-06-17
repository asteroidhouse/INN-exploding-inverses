import argparse
import os
import json
import shutil
import random
from itertools import islice
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss
from torchvision.utils import make_grid

from datasets import get_CIFAR10, get_SVHN, get_MNIST, postprocess
from model import Glow, InvDiscriminator
import mine
import matplotlib.pyplot as plt
import ipdb
from utils import uniform_binning_correction
from recon_mnist import run_recon_evolution
from inception_score import inception_score, run_fid
from csv_logger import CSVLogger, plot_csv
import utils
import cgan_models
device = 'cpu' if (not torch.cuda.is_available()) else 'cuda:0'
nn = torch.nn
import numpy as np

class DCGANDiscriminator(nn.Module):
    def __init__(self, imgSize, ndf, nc):
        super(DCGANDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(input.size(0), -1).mean(-1, keepdim=True)

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print('Using seed: {seed}'.format(seed=seed))

def gradient_penalty(x, y, f):
    """From https://github.com/LynnHo/Pytorch-WGAN-GP-DRAGAN-Celeba/blob/master/train_celeba_wgan_gp.py
    """
    # Interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape, device='cuda:0', requires_grad=True)
    z = x + alpha * (y - x)

    # Gradient penalty
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size(), device=x.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp


def check_dataset(dataset, dataroot, augment, download):
    if dataset == 'cifar10':
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == 'svhn':
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    if dataset == 'mnist':
        mnist = get_MNIST(False, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = mnist

    return input_size, num_classes, train_dataset, test_dataset


def compute_loss(nll, reduction='mean'):
    if reduction == 'mean':
        losses = {'nll': torch.mean(nll)}
    elif reduction == 'none':
        losses = {'nll': nll}

    losses['total_loss'] = losses['nll']

    return losses

def compute_jacobian_regularizer(x, z, n_proj=1):
    """
        x: A minibatch of examples
        z: Model outputs
        n_proj: number of random projections
    """
    x_flat = x.view(x.size(0), -1)  # (128, 3072)
    z_flat = z.view(z.size(0), -1)  # (128, 3072)
    batch_size, C = x_flat.size()  # batch_size=128, C=3072
    J_f = 0

    for i in range(n_proj):
        v_c = torch.randn(size=(batch_size, C), device=x.device)  # (128, 3072)
        v = v_c / (torch.norm(v_c, dim=1).unsqueeze(1) + 1e-6)


        z_vec = z.view(-1)
        v_vec = v.view(-1)

        Jv = torch.autograd.grad(torch.matmul(z_vec, v_vec), x, create_graph=True)[0]
        # J_f += torch.norm(Jv)**2
        J_f += torch.norm(Jv)

    return J_f

def compute_jacobian_regularizer_manyinputs(xs, z, n_proj=1):
    """
        x: A minibatch of examples
        z: Model outputs
        n_proj: number of random projections
    """
    J_f = 0
    for x in xs:
        z_flat = z.view(z.size(0), -1)  # (128, large)
        for i in range(n_proj):
            v_c = torch.randn(size=z_flat.size(), device=x.device)
            v = v_c / (torch.norm(v_c, dim=1).unsqueeze(1) + 1e-6)
            z_vec = z.view(-1)
            v_vec = v.view(-1)

            Jv = torch.autograd.grad(torch.matmul(z_vec, v_vec), x, create_graph=True)[0]
            # Jv = torch.autograd.grad(torch.matmul(z_vec, v_vec), x, create_graph=True)[0]
            # J_f += torch.norm(Jv)**2
            J_f += torch.norm(Jv)
    return J_f

def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction='mean'):
    if reduction == 'mean':
        losses = {'nll': torch.mean(nll)}
    elif reduction == 'none':
        losses = {'nll': nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(y_logits,
                                                          y,
                                                          reduction=reduction)
    else:
        loss_classes = F.cross_entropy(y_logits,
                                       torch.argmax(y, dim=1),
                                       reduction=reduction)

    losses['loss_classes'] = loss_classes
    losses['total_loss'] = losses['nll'] + y_weight * loss_classes

    return losses


def cycle(loader):
    while True:
        for data in loader:
            yield data

def generate_from_noise(model, batch_size,clamp=False, guard_nans=True):
    assert not clamp
    _, c2, h, w  = model.prior_h.shape
    c = c2 // 2
    zshape = (batch_size, c, h, w)
    randz  = torch.randn(zshape).to(device)
    randz  = torch.autograd.Variable(randz, requires_grad=True)
    if clamp:
        randz = torch.clamp(randz,-5,5)
    images = model(z= randz, y_onehot=None, temperature=1, reverse=True,batch_size=0)
    if guard_nans:
        images[(images!=images)] = 0
    return images

def main(dataset, dataroot, download, augment, batch_size, eval_batch_size,
         epochs, saved_model, seed, hidden_channels, K, L, actnorm_scale,
         flow_permutation, flow_coupling, LU_decomposed, learn_top,
         y_condition, y_weight, max_grad_clip, max_grad_norm, lr,
         n_workers, cuda, n_init_batches, warmup_steps, output_dir,
         saved_optimizer, warmup, fresh,logittransform, gan, disc_lr,sn,flowgan, eval_every, ld_on_samples, weight_gan, weight_prior,weight_logdet, jac_reg_lambda,affine_eps, no_warm_up, optim_name,clamp, svd_every, eval_only,no_actnorm,affine_scale_eps,actnorm_max_scale, no_conv_actnorm,affine_max_scale,actnorm_eps,init_sample,no_split, disc_arch, weight_entropy_reg,db):


    check_manual_seed(seed)

    ds = check_dataset(dataset, dataroot, augment, download)
    image_shape, num_classes, train_dataset, test_dataset = ds

    # Note: unsupported for now
    multi_class = False

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=n_workers,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size,
                                  shuffle=False, num_workers=n_workers,
                                  drop_last=False)
    model = Glow(image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, num_classes,
                 learn_top, y_condition,logittransform,sn,affine_eps,no_actnorm,affine_scale_eps, actnorm_max_scale, no_conv_actnorm,affine_max_scale,actnorm_eps,no_split)

    model = model.to(device)

    if disc_arch == 'mine':
        discriminator = mine.Discriminator(image_shape[-1])
    elif disc_arch == 'biggan':
        discriminator = cgan_models.Discriminator(image_channels=image_shape[-1],conditional_D=False)
    elif disc_arch == 'dcgan':
        discriminator = DCGANDiscriminator(image_shape[0], 64, image_shape[-1])
    elif disc_arch == 'inv':
        discriminator = InvDiscriminator(image_shape, hidden_channels, K, L, actnorm_scale,
                 flow_permutation, flow_coupling, LU_decomposed, num_classes,
                 learn_top, y_condition,logittransform,sn,affine_eps,no_actnorm,affine_scale_eps, actnorm_max_scale, no_conv_actnorm,affine_max_scale,actnorm_eps,no_split)

    discriminator = discriminator.to(device)
    D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=disc_lr, betas=(.5, .99), weight_decay=0)
    if optim_name =='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.5, .99), weight_decay=0)
    elif optim_name=='adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    if not no_warm_up:
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    iteration_fieldnames = ['global_iteration', 'fid','sample_pad', 'train_bpd','eval_bpd', 'pad','batch_real_acc','batch_fake_acc','batch_acc']
    iteration_logger = CSVLogger(fieldnames=iteration_fieldnames,
                             filename=os.path.join(output_dir, 'iteration_log.csv'))
    iteration_fieldnames = ['global_iteration'
                            ,'condition_num'
                            ,'max_sv'
                            ,'min_sv'
                            ,'inverse_condition_num'
                            ,'inverse_max_sv'
                            ,'inverse_min_sv']
    svd_logger = CSVLogger(fieldnames=iteration_fieldnames,
                             filename=os.path.join(output_dir, 'svd_log.csv'))

    #
    test_iter = test_loader.__iter__()
    N_inception = 1000
    x_real_inception = torch.cat([test_iter.__next__()[0].to(device) for _ in range(N_inception//args.batch_size+1)],0 )[:N_inception]
    x_real_inception = x_real_inception + .5
    x_for_recon = test_iter.__next__()[0].to(device)

    def gan_step(engine, batch):
        assert not y_condition
        if 'iter_ind' in dir(engine):
            engine.iter_ind += 1
        else:
            engine.iter_ind = -1
        losses = {}
        model.train()
        discriminator.train()


        x, y = batch
        x = x.to(device)


        def run_noised_disc(discriminator, x):
            x = uniform_binning_correction(x)[0]
            return discriminator(x)

        real_acc = fake_acc = acc = 0
        if weight_gan > 0:
            fake = generate_from_noise(model, x.size(0), clamp=clamp)

            D_real_scores = run_noised_disc(discriminator, x.detach())
            D_fake_scores = run_noised_disc(discriminator, fake.detach())

            ones_target = torch.ones((x.size(0), 1), device=x.device)
            zeros_target = torch.zeros((x.size(0), 1), device=x.device)

            D_real_accuracy = torch.sum(torch.round(F.sigmoid(D_real_scores)) == ones_target).float() / ones_target.size(0)
            D_fake_accuracy = torch.sum(torch.round(F.sigmoid(D_fake_scores)) == zeros_target).float() / zeros_target.size(0)

            D_real_loss = F.binary_cross_entropy_with_logits(D_real_scores, ones_target)
            D_fake_loss = F.binary_cross_entropy_with_logits(D_fake_scores, zeros_target)

            D_loss = (D_real_loss + D_fake_loss) / 2
            gp = gradient_penalty(x.detach(), fake.detach(), lambda _x: run_noised_disc(discriminator, _x))
            D_loss_plus_gp = D_loss  + 10*gp
            D_optimizer.zero_grad()
            D_loss_plus_gp.backward()
            D_optimizer.step()


            # Train generator
            fake = generate_from_noise(model, x.size(0), clamp=clamp, guard_nans=False)
            G_loss = F.binary_cross_entropy_with_logits(run_noised_disc(discriminator, fake), torch.ones((x.size(0), 1), device=x.device))

            # Trace
            real_acc = D_real_accuracy.item()
            fake_acc = D_fake_accuracy.item()
            acc = .5*(D_fake_accuracy.item()+D_real_accuracy.item())

        z, nll, y_logits, (prior, logdet)= model.forward(x, None, return_details=True)
        train_bpd = nll.mean().item()

        loss = 0
        if weight_gan > 0:
            loss = loss +  weight_gan * G_loss
        if weight_prior > 0:
            loss = loss +  weight_prior * -prior.mean()
        if weight_logdet > 0:
            loss = loss + weight_logdet * -logdet.mean()

        if weight_entropy_reg > 0:
            _, _, _, (sample_prior, sample_logdet)= model.forward(fake, None, return_details=True)
            # notice this is actually "decreasing" sample likelihood.
            loss = loss + weight_entropy_reg * (sample_prior.mean() + sample_logdet.mean())
        # Jac Reg
        if jac_reg_lambda > 0:
            # Sample
            x_samples = generate_from_noise(model, args.batch_size, clamp=clamp).detach()
            x_samples.requires_grad_()
            z = model.forward(x_samples, None, return_details=True)[0]
            other_zs = torch.cat([split._last_z2.view(x.size(0),-1) for  split  in model.flow.splits],-1)
            all_z = torch.cat([other_zs, z.view(x.size(0),-1)], -1)
            sample_foward_jac = compute_jacobian_regularizer(x_samples, all_z, n_proj=1)
            _, c2, h, w  = model.prior_h.shape
            c = c2 // 2
            zshape = (batch_size, c, h, w)
            randz  = torch.randn(zshape).to(device)
            randz  = torch.autograd.Variable(randz, requires_grad=True)
            images = model(z= randz, y_onehot=None, temperature=1, reverse=True,batch_size=0)
            other_zs = [split._last_z2 for  split  in model.flow.splits]
            all_z = [randz] + other_zs
            sample_inverse_jac = compute_jacobian_regularizer_manyinputs(all_z, images, n_proj=1)

            # Data
            x.requires_grad_()
            z = model.forward(x, None, return_details=True)[0]
            other_zs = torch.cat([split._last_z2.view(x.size(0),-1) for  split  in model.flow.splits],-1)
            all_z = torch.cat([other_zs, z.view(x.size(0),-1)], -1)
            data_foward_jac = compute_jacobian_regularizer(x, all_z, n_proj=1)
            _, c2, h, w  = model.prior_h.shape
            c = c2 // 2
            zshape = (batch_size, c, h, w)
            z.requires_grad_()
            images = model(z=z, y_onehot=None, temperature=1, reverse=True,batch_size=0)
            other_zs = [split._last_z2 for  split  in model.flow.splits]
            all_z = [z] + other_zs
            data_inverse_jac = compute_jacobian_regularizer_manyinputs(all_z, images, n_proj=1)

            # loss = loss + jac_reg_lambda * (sample_foward_jac + sample_inverse_jac )
            loss = loss + jac_reg_lambda * (sample_foward_jac + sample_inverse_jac  +data_foward_jac  + data_inverse_jac)

        if not eval_only:
            optimizer.zero_grad()
            loss.backward()
            if not db:
                assert max_grad_clip == max_grad_norm == 0
            if max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Replace NaN gradient with 0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    g = p.grad.data
                    g[g!=g] = 0

            optimizer.step()

        if engine.iter_ind  % 100==0:
            with torch.no_grad():
                fake = generate_from_noise(model, x.size(0), clamp=clamp)
                z = model.forward(fake, None, return_details=True)[0]
            print("Z max min")
            print(z.max().item(), z.min().item())
            if (fake!=fake).float().sum() > 0:
                title='NaNs'
            else:
                title="Good"
            grid = make_grid((postprocess(fake.detach().cpu(), dataset)[:30]), nrow=6).permute(1,2,0)
            plt.figure(figsize=(10,10))
            plt.imshow(grid)
            plt.axis('off')
            plt.title(title)
            plt.savefig(os.path.join(output_dir, f'sample_{engine.iter_ind}.png'))

        if engine.iter_ind  % eval_every==0:
            torch.save(model, os.path.join(output_dir, f'ckpt_{engine.iter_ind}.pt'))

            model.eval()

            with torch.no_grad():
                # Plot recon
                fpath = os.path.join(output_dir, '_recon', f'recon_{engine.iter_ind}.png')
                sample_pad = run_recon_evolution(model, generate_from_noise(model, args.batch_size, clamp=clamp).detach(), fpath)
                print(f"Iter: {engine.iter_ind}, Recon Sample PAD: {sample_pad}")

                pad = run_recon_evolution(model, x_for_recon, fpath)
                print(f"Iter: {engine.iter_ind}, Recon PAD: {pad}")
                pad =  pad.item()
                sample_pad  =  sample_pad.item()

                # Inception score
                sample = torch.cat([generate_from_noise(model, args.batch_size, clamp=clamp) for _ in range(N_inception//args.batch_size+1)],0 )[:N_inception]
                sample = sample + .5

                if (sample!=sample).float().sum() > 0:
                    print("Sample NaNs")
                    raise
                else:
                    fid =  run_fid(x_real_inception.clamp_(0,1),sample.clamp_(0,1) )
                    print(f'fid: {fid}, global_iter: {engine.iter_ind}')

                # Eval BPD
                eval_bpd = np.mean([model.forward(x.to(device), None, return_details=True)[1].mean().item() for x, _ in test_loader])

                stats_dict = {
                        'global_iteration': engine.iter_ind ,
                        'fid': fid,
                        'train_bpd': train_bpd,
                        'pad': pad,
                        'eval_bpd': eval_bpd,
                        'sample_pad': sample_pad,
                        'batch_real_acc':real_acc,
                        'batch_fake_acc':fake_acc,
                        'batch_acc':acc
                }
                iteration_logger.writerow(stats_dict)
                plot_csv(iteration_logger.filename)
            model.train()

        if  engine.iter_ind + 2 % svd_every == 0:
            model.eval()
            svd_dict = {}
            ret = utils.computeSVDjacobian(x_for_recon, model)
            D_for, D_inv = ret['D_for'], ret['D_inv']
            cn = float(D_for.max()/ D_for.min())
            cn_inv = float(D_inv.max()/ D_inv.min())
            svd_dict['global_iteration'] =  engine.iter_ind
            svd_dict['condition_num'] = cn
            svd_dict['max_sv'] = float(D_for.max())
            svd_dict['min_sv'] = float(D_for.min())
            svd_dict['inverse_condition_num'] = cn_inv
            svd_dict['inverse_max_sv'] = float(D_inv.max())
            svd_dict['inverse_min_sv'] = float(D_inv.min())
            svd_logger.writerow(svd_dict)
            model.train()
            if eval_only:
                sys.exit()

        # Dummy
        losses['total_loss'] = torch.mean(nll).item()
        return losses


    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(nll, y_logits, y_weight, y,
                                        multi_class, reduction='none')
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction='none')

        return losses
    trainer = Engine(gan_step)
    # else:
    #     trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, 'glow', save_interval=5,
                                         n_saved=1, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
                              {'model': model, 'optimizer': optimizer})

    monitoring_metrics = ['total_loss']
    RunningAverage(output_transform=lambda x: x['total_loss']).attach(trainer, 'total_loss')

    evaluator = Engine(eval_step)

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(lambda x, y: torch.mean(x), output_transform=lambda x: (x['total_loss'], torch.empty(x['total_loss'].shape[0]))).attach(evaluator, 'total_loss')

    if y_condition:
        monitoring_metrics.extend(['nll'])
        RunningAverage(output_transform=lambda x: x['nll']).attach(trainer, 'nll')

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(lambda x, y: torch.mean(x), output_transform=lambda x: (x['nll'], torch.empty(x['nll'].shape[0]))).attach(evaluator, 'nll')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if given
    if saved_model:
        print("Loading...")
        print(saved_model)
        loaded =  torch.load(saved_model)
        # if 'Glow' in str(type(loaded)):
        #     model  = loaded
        # else:
        #     raise
        # # if 'Glow' in str(type(loaded)):
        # #     loaded  = loaded.state_dict()
        model.load_state_dict(loaded)
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split('_')[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)


    @trainer.on(Events.STARTED)
    def init(engine):
        if saved_model:
            return
        model.train()
        print("Initializing Actnorm...")
        init_batches = []
        init_targets = []

        if n_init_batches == 0:
            model.set_actnorm_init()
            return
        with torch.no_grad():
            if init_sample:
                generate_from_noise(model, args.batch_size * args.n_init_batches)
            else:
                for batch, target in islice(train_loader, None,
                                            n_init_batches):
                    init_batches.append(batch)
                    init_targets.append(target)

                init_batches = torch.cat(init_batches).to(device)

                assert init_batches.shape[0] == n_init_batches * batch_size

                if y_condition:
                    init_targets = torch.cat(init_targets).to(device)
                else:
                    init_targets = None

                model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)
        if not no_warm_up:
            scheduler.step()
        metrics = evaluator.state.metrics

        losses = ', '.join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f'Validation Results - Epoch: {engine.state.epoch} {losses}')

    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f'Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]')
        timer.reset()

    trainer.run(train_loader, epochs)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='cifar10', choices=['cifar10', 'svhn', 'mnist'],
                        help='Type of the dataset to be used.')

    parser.add_argument('--dataroot',
                        type=str, default='./data',
                        help='path to dataset')

    parser.add_argument('--download', default=True)

    parser.add_argument('--no_augment', action='store_false',
                        dest='augment', help='Augment training data')

    parser.add_argument('--hidden_channels',
                        type=int, default=512,
                        help='Number of hidden channels')

    parser.add_argument('--K',
                        type=int, default=32,
                        help='Number of layers per block')

    parser.add_argument('--L',
                        type=int, default=3,
                        help='Number of blocks')

    parser.add_argument('--actnorm_scale',
                        type=float, default=1.0,
                        help='Act norm scale')

    parser.add_argument('--flow_permutation', type=str,
                        default='invconv', choices=['invconv', 'shuffle', 'reverse'],
                        help='Type of flow permutation')

    parser.add_argument('--flow_coupling', type=str,
                        default='affine', choices=['additive', 'affine','naffine','gaffine'],
                        help='Type of flow coupling')

    parser.add_argument('--no_LU_decomposed', action='store_false',
                        dest='LU_decomposed',
                        help='Train with LU decomposed 1x1 convs')

    parser.add_argument('--no_learn_top', action='store_false',
                        help='Do not train top layer (prior)', dest='learn_top')

    parser.add_argument('--y_condition', action='store_true',
                        help='Train using class condition')

    parser.add_argument('--y_weight',
                        type=float, default=0.01,
                        help='Weight for class condition loss')

    parser.add_argument('--max_grad_clip',
                        type=float, default=0,
                        help='Max gradient value (clip above - for off)')

    parser.add_argument('--max_grad_norm',
                        type=float, default=0,
                        help='Max norm of gradient (clip above - 0 for off)')

    parser.add_argument('--n_workers',
                        type=int, default=6,
                        help='number of data loading workers')

    parser.add_argument('--batch_size',
                        type=int, default=64,
                        help='batch size used during training')

    parser.add_argument('--eval_batch_size',
                        type=int, default=512,
                        help='batch size used during evaluation')

    parser.add_argument('--epochs',
                        type=int, default=250,
                        help='number of epochs to train for')

    parser.add_argument('--lr',
                        type=float, default=5e-4,
                        help='initial learning rate')

    parser.add_argument('--warmup',
                        type=float, default=5,
                        help='Warmup learning rate linearly per epoch')

    parser.add_argument('--warmup_steps',
                        type=int, default=4000,
                        help='Number of warmup steps for lr initialisation')

    parser.add_argument('--n_init_batches',
                        type=int, default=8,
                        help='Number of batches to use for Act Norm initialisation')

    parser.add_argument('--no_cuda',
                        action='store_false',
                        dest='cuda',
                        help='disables cuda')

    parser.add_argument('--output_dir',
                        default='.',
                        help='directory to output logs and model checkpoints')

    parser.add_argument('--fresh',
                        action='store_true',
                        help='Remove output directory before starting')

    parser.add_argument('--saved_model',
                        default='',
                        help='Path to model to load for continuing training')

    parser.add_argument('--saved_optimizer',
                        default='',
                        help='Path to optimizer to load for continuing training')

    parser.add_argument('--seed',
                        type=int, default=0,
                        help='manual seed')
    parser.add_argument('--logittransform',
                       type=int, default=0)
    parser.add_argument('--gan',
                        action='store_true')
    parser.add_argument('--sn', type=int, default=0)
    parser.add_argument('--flowgan', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--weight_gan', type=float, default=0)
    parser.add_argument('--ld_on_samples', type=int, default=0)
    parser.add_argument('--weight_prior', type=float, default=0)
    parser.add_argument('--weight_logdet', type=float, default=0)
    parser.add_argument('--weight_entropy_reg', type=float, default=0)
    parser.add_argument('--jac_reg_lambda', type=float, default=0)
    parser.add_argument('--affine_eps', type=float, default=0)
    parser.add_argument('--disc_lr',type=float, default=1e-5)
    parser.add_argument('--no_warm_up',type=int, default=0)
    parser.add_argument('--optim_name',type=str, default='adam')
    parser.add_argument('--clamp',type=int, default=0)
    parser.add_argument('--svd_every',type=int, default=1)
    parser.add_argument('--eval_only',type=int, default=0)
    parser.add_argument('--no_actnorm',type=int, default=0)
    parser.add_argument('--affine_scale_eps',type=float, default=0)
    parser.add_argument('--actnorm_max_scale',type=float, default=0)
    parser.add_argument('--no_conv_actnorm',type=int, default=0)
    parser.add_argument('--affine_max_scale',type=float, default=0)
    parser.add_argument('--actnorm_eps',type=float, default=0)
    parser.add_argument('--init_sample',type=int, default=0)
    parser.add_argument('--no_split',type=int, default=0)
    parser.add_argument('--disc_arch', type=str, default='mine', choices=['dcgan','mine','inv','biggan'])
    parser.add_argument('--db', type=int, default=0)

    args = parser.parse_args()
    kwargs = vars(args)

    makedirs(args.dataroot)

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (len(os.listdir(args.output_dir)) > 0):
            raise FileExistsError("Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag.")
    os.makedirs(os.path.join(args.output_dir, '_recon'))

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    log_file = os.path.join(args.output_dir, 'log.txt')
    log = open(log_file, 'w')
    _print = print
    def print(*content):
        _print(*content)
        _print(*content, file=log)
        log.flush()

    main(**kwargs)
    log.close()
