"""Measure the angle between the true gradient (computed using stored activations) and
the memory-saving gradient (computed using reconstructed activations).

Example
-------
python compute_gradient_angles.py --load=saves/affine_conv/EXPERIMENT_FILE_NAME
"""
import os
import sys
import ipdb
import time
import random
import datetime
import argparse
import pickle as pkl
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad

from torchvision import transforms

# Local imports
import utils
import data_loaders
from invertible_layers2_mem import *


# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# Training
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--n_bits_x', type=int, default=8.,
                    help='')
parser.add_argument('--zs_dim', type=int, default=3072,
                    help='The number of dimensions to use for z_s (the remaining D - zs_dim become z_n)')
parser.add_argument('--load', type=str, default='',
                    help='Path to checkpoint to load')

# Logging
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()

args.n_bins = 2 ** args.n_bits_x

use_device = 'cuda:0'

# Set random seed for reproducibility
if args.seed is not None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


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

channels = 3
imsize = 32
num_classes = 10

train_loader, test_loader, init_loader = data_loaders.load_cifar10(args.batch_size, tf, tf_test, shuffle=False)

def run_flow(img):
    objective = torch.zeros_like(img[:, 0, 0, 0])
    objective += float(-np.log(args.n_bins) * np.prod(img.shape[1:]))
    return model(img, objective)


if args.load == '':
    print('You need to pass in a checkpoint to load with --load')
    sys.exit(0)

load = args.load

checkpoints = [fname for fname in os.listdir(load) if fname.startswith('model_ep')]
checkpoints = sorted(checkpoints, key=lambda x: int(x.split('iter_')[1].split('.')[0]))
checkpoints.insert(0, 'model_initialization.pt')

angles = defaultdict(list)
l2_dists = defaultdict(list)
normalized_l2_dists = defaultdict(list)

true_gradients = defaultdict(list)
memsave_gradients = defaultdict(list)

true_gradient_lists = defaultdict(list)
memsave_gradient_lists = defaultdict(list)

for (i, fname) in enumerate(checkpoints):
    if fname == 'model_initialization.pt':
        fname_iteration = 0
    else:
        fname_iteration = int(fname.split('iter_')[1].split('.')[0])
    print('{}, Iteration {}'.format(fname, fname_iteration))
    sys.stdout.flush()

    # Load model from checkpoint
    model = torch.load(os.path.join(load, fname))
    projection = torch.load(os.path.join(load, fname.replace('model', 'projection')))

    model.eval()
    projection.eval()

    params = list(model.parameters()) + list(projection.parameters())
    optimizer = optim.SGD(params, lr=0, weight_decay=0)

    for idx, (img, label) in enumerate(tqdm(train_loader)):
        img, label = img.to(use_device), label.to(use_device)

        # TRUE GRADIENTS
        # --------------------
        z, objective = run_flow(img)
        zs = z.view(z.size(0), -1)[:, :args.zs_dim]

        logits = projection(zs)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        true_gradient_list = [p.grad.clone().view(-1) + 1e-8 for p in model.parameters()]
        true_gradient = torch.cat(true_gradient_list)
        optimizer.zero_grad()
        # --------------------

        # MEMORY-SAVING GRADIENTS
        # --------------------
        z, objective = run_flow(img)
        zs = z.view(z.size(0), -1)[:, :args.zs_dim]

        logits = projection(zs)
        loss = F.cross_entropy(logits, label)

        z_grad = torch.autograd.grad(loss, z)[0]
        model_gradient = model.module.gradient(z, z_grad)
        model_gradient = [g for g in model_gradient if g is not None]

        memsave_gradient_list = [g.view(-1) + 1e-8 for g in model_gradient]
        memsave_gradient = torch.cat(memsave_gradient_list)
        # --------------------

        if bool(int(torch.any(torch.isnan(memsave_gradient)))) or bool(int(torch.any(torch.isinf(memsave_gradient)))):
            angle = float('nan')
        else:
            angle = utils.angle_between(true_gradient, memsave_gradient).item()
        l2 = torch.norm(true_gradient - memsave_gradient).item()

        angles[fname_iteration].append(angle)
        l2_dists[fname_iteration].append(l2)
        normalized_l2_dists[fname_iteration].append(l2 / torch.norm(true_gradient).item())

        if idx > 40:
            break

    print(angles)
    sys.stdout.flush()

    with open(os.path.join(load, 'grad_comparison.pkl'), 'wb') as f:
        pkl.dump({ 'angles': angles, 'l2_dists': l2_dists, 'normalized_l2_dists': normalized_l2_dists }, f)


with open(os.path.join(load, 'grad_comparison.pkl'), 'wb') as f:
    pkl.dump({ 'angles': angles, 'l2_dists': l2_dists }, f)

print('Saved!')
