"""Evaluate the reconstruction error of a pre-trained Residual Flow on OOD data.

Example
-------
python resflow_ood_eval.py
"""
import os
import gc
import sys
import ipdb
import time
import math
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import lib.utils as utils
import lib.layers as layers
import lib.optimizers as optim
import lib.layers.base as base_layers
from lib.resflow import ACT_FNS, ResidualFlow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10', choices=['cifar10', 'imagenet32', 'imagenet64',])
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)

parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--wd', help='Weight decay', type=float, default=0)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--annealing-iters', type=int, default=0)
parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--update-freq', type=int, default=1)

parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--begin-epoch', type=int, default=0)

parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)


def standard_normal_sample(size):
    return torch.randn(size)


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x


logger.info('Loading dataset {}'.format(args.data))
# Dataset and hyperparameters
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 10
    if args.task in ['classification', 'hybrid']:
        # Remove the logit transform.
        init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        init_layer = layers.LogitTransform(0.05)

logger.info('Dataset loaded.')
logger.info('Creating model.')

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)
# dataset_size = len(train_loader.dataset)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)


# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification=args.task in ['classification', 'hybrid'],
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)

model.to(device)
logger.info(model)

best_test_bpd = math.inf

args.resume = 'cifar10_resflow_16-16-16.pth'

if args.resume is not None:
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    checkpt = torch.load(args.resume)
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)
    # ema.set(checkpt['ema'])
    if 'optimizer_state_dict' in checkpt:
        optimizer.load_state_dict(checkpt['optimizer_state_dict'])
        # Manually move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    del checkpt
    del state

model.eval()

fixed_z = standard_normal_sample([min(32, args.batchsize), (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

utils.makedirs(os.path.join(args.save, 'imgs'))
nvals = 256

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')

def save_samples(images, name):
    with open(os.path.join('reconstructions', '{}.pkl'.format(name)), 'wb') as f:
        pkl.dump(images, f)

    images = images
    grid = make_grid(images[:16], 4, padding=1)  # Should be a 4x4 grid
    save_image(grid, '{}.png'.format(name))


def compute_total_recons_error(dataloader, name):
    all_recons_errors = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.cuda()
            z, logpx = model(images, 0)
            recons = model(z, inverse=True)

            minibatch_recons_errs = [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, images)]
            all_recons_errors += minibatch_recons_errs

            mean_error, min_error, max_error = np.mean(minibatch_recons_errs), np.min(minibatch_recons_errs), np.max(minibatch_recons_errs)
            print('{} {} | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(name, i, mean_error, min_error, max_error))

            save_samples(images, '{}_orig_{}'.format(name, i))
            save_samples(recons, '{}_recons_{}'.format(name, i))

    all_recons_errors = np.array(all_recons_errors)
    return all_recons_errors


transform = transforms.Compose([transforms.ToTensor()])

train_cifar = datasets.CIFAR10(root='../data/CIFAR10', download=True, train=True, transform=transform)
test_cifar = datasets.CIFAR10(root='../data/CIFAR10', download=True, train=False, transform=transform)

train_svhn = datasets.SVHN(root='../data/SVHN', download=True, split='train', transform=transform)
test_svhn = datasets.SVHN(root='../data/SVHN', download=True, split='test', transform=transform)

# The data is in the range [0, 1]
train_dataloader_cifar = torch.utils.data.DataLoader(train_cifar, batch_size=32, num_workers=0, pin_memory=True)
test_dataloader_cifar = torch.utils.data.DataLoader(test_cifar, batch_size=32, num_workers=0, pin_memory=True)

# The data is in the range [0, 1]
train_dataloader_svhn = torch.utils.data.DataLoader(train_svhn, batch_size=32, num_workers=0, pin_memory=True)
test_dataloader_svhn = torch.utils.data.DataLoader(test_svhn, batch_size=32, num_workers=0, pin_memory=True)

minibatch_size = 25
num_batches = 400

# Gaussian
gaussian_recons_errs = []
for batch_num in range(num_batches):
    print('Gaussian batch {}'.format(batch_num))
    with torch.no_grad():
        gaussian_data = np.clip(.5 + np.random.normal(size=(minibatch_size, 3, 32, 32)), a_min=0.01, a_max=0.99)
        gaussian_data = torch.from_numpy(gaussian_data).float()
        gaussian_data = gaussian_data.cuda()
        z, logpx = model(gaussian_data, 0)
        recons = model(z, inverse=True)

    gaussian_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, gaussian_data)]
    save_samples(gaussian_data, 'gaussian_orig')
    save_samples(recons, 'gaussian_recons')

print('Gaussian rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(gaussian_recons_errs), np.min(gaussian_recons_errs), np.max(gaussian_recons_errs)))
sys.stdout.flush()

# Uniform
uniform_recons_errs = []
for batch_num in range(num_batches):
    print('Uniform batch {}'.format(batch_num))
    with torch.no_grad():
        uniform_data = np.clip(np.random.uniform(size=(minibatch_size, 3, 32, 32)), a_min=0.01, a_max=0.99)
        uniform_data = torch.from_numpy(uniform_data).float()
        uniform_data = uniform_data.cuda()
        z, logpx = model(uniform_data, 0)
        recons = model(z, inverse=True)

    uniform_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, uniform_data)]
    save_samples(uniform_data, 'uniform_orig_{}'.format(batch_num))
    save_samples(recons, 'uniform_recons_{}'.format(batch_num))

print('Uniform rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(uniform_recons_errs), np.min(uniform_recons_errs), np.max(uniform_recons_errs)))
sys.stdout.flush()

# Rademacher
rademacher_recons_errs = []
for batch_num in range(num_batches):
    print('Rademacher batch {}'.format(batch_num))
    with torch.no_grad():
        rademacher_data = np.random.binomial(1, .5, size=(minibatch_size, 3, 32, 32))
        rademacher_data = torch.from_numpy(rademacher_data).float()
        rademacher_data = rademacher_data.cuda()
        z, logpx = model(rademacher_data, 0)
        recons = model(z, inverse=True)

    rademacher_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, rademacher_data)]
    save_samples(rademacher_data, 'rademacher_orig_{}'.format(batch_num))
    save_samples(recons, 'rademacher_recons_{}'.format(batch_num))

print('Rademacher rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(rademacher_recons_errs), np.min(rademacher_recons_errs), np.max(rademacher_recons_errs)))
sys.stdout.flush()

# Texture OOD Data
texture_recons_errs = []
for batch_num in range(num_batches):
    print('Texture batch {}'.format(batch_num))
    with torch.no_grad():
        texture_data = torch.load(os.path.join('../ood_data/dtd.t7')).numpy() / 255.0
        texture_data = torch.from_numpy(texture_data).float()
        texture_data = texture_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size].cuda()  # Just taking one minibatch here
        z, logpx = model(texture_data, 0)
        recons = model(z, inverse=True)

    texture_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, texture_data)]
    save_samples(texture_data, 'texture_orig_{}'.format(batch_num))
    save_samples(recons, 'texture_recons_{}'.format(batch_num))

print('Texture rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(texture_recons_errs), np.min(texture_recons_errs), np.max(texture_recons_errs)))
sys.stdout.flush()

# Places OOD Data
places_recons_errs = []
for batch_num in range(num_batches):
    print('Places batch {}'.format(batch_num))
    with torch.no_grad():
        places_data = torch.load(os.path.join('../ood_data/places.t7')).numpy() / 255.
        places_data = torch.from_numpy(places_data).float()
        places_data = places_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size].cuda()  # Just taking one minibatch here
        z, logpx = model(places_data, 0)
        recons = model(z, inverse=True)

    places_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, places_data)]
    save_samples(places_data, 'places_orig_{}'.format(batch_num))
    save_samples(recons, 'places_recons_{}'.format(batch_num))

print('Places rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(places_recons_errs), np.min(places_recons_errs), np.max(places_recons_errs)))
sys.stdout.flush()


tim_resize_data = datasets.ImageFolder('../ood_data/Imagenet_resize', transform=transform)
tim_resize_data = np.stack([img.numpy() for (img, label) in tim_resize_data])

tim_resized_recons_errs = []
for batch_num in range(num_batches):
    print('tinyImageNet batch {}'.format(batch_num))
    tim_resize_minibatch = torch.from_numpy(tim_resize_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size]).float()  # Range is [0, 1]
    tim_resize_minibatch = tim_resize_minibatch.cuda()

    with torch.no_grad():
        z, logpx = model(tim_resize_minibatch, 0)
        recons = model(z, inverse=True)

    tim_resized_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, tim_resize_minibatch)]
    save_samples(tim_resize_minibatch, 'tim_resized_orig_{}'.format(batch_num))
    save_samples(recons, 'tim_resized_recons_{}'.format(batch_num))

print('tinyImageNet-resized rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(tim_resized_recons_errs), np.min(tim_resized_recons_errs), np.max(tim_resized_recons_errs)))
sys.stdout.flush()


print('Computing recons error for CIFAR train')
train_cifar_errors = compute_total_recons_error(train_dataloader_cifar, 'cifar_train')
mean_train_cifar, min_train_cifar, max_train_cifar = np.mean(train_cifar_errors), np.min(train_cifar_errors), np.max(train_cifar_errors)
print('CIFAR train | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(mean_train_cifar, min_train_cifar, max_train_cifar))

print('Computing recons error for CIFAR test')
test_cifar_errors = compute_total_recons_error(test_dataloader_cifar, 'cifar_test')
mean_test_cifar, min_test_cifar, max_test_cifar = np.mean(test_cifar_errors), np.min(test_cifar_errors), np.max(test_cifar_errors)
print('CIFAR test | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(mean_test_cifar, min_test_cifar, max_test_cifar))

print('Computing recons error for SVHN train')
train_svhn_errors = compute_total_recons_error(train_dataloader_svhn, 'svhn_train')
mean_train_svhn, min_train_svhn, max_train_svhn = np.mean(train_svhn_errors), np.min(train_svhn_errors), np.max(train_svhn_errors)
print('SVHN train | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(mean_train_svhn, min_train_svhn, max_train_svhn))

print('Computing recons error for SVHN test')
test_svhn_errors = compute_total_recons_error(test_dataloader_svhn, 'svhn_test')
mean_test_svhn, min_test_svhn, max_test_svhn = np.mean(test_svhn_errors), np.min(test_svhn_errors), np.max(test_svhn_errors)
print('SVHN test | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(mean_test_svhn, min_test_svhn, max_test_svhn))
sys.stdout.flush()
