"""Evaluate the reconstruction error of a pre-trained Glow model on OOD data.

Example
-------
python glow_ood_eval.py
"""
import os
import sys
import ipdb
import json
import pickle as pkl
from tqdm import tqdm

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision

# Local imports
from model import Glow
from datasets import get_CIFAR10, get_SVHN


device = torch.device("cuda")

output_folder = 'pretrained/'
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:
    hparams = json.load(json_file)

print(hparams)

image_shape, num_classes, train_cifar, test_cifar = get_CIFAR10(augment=False, dataroot=hparams['dataroot'], download=True)
image_shape, num_classes_svhn, train_svhn, test_svhn = get_SVHN(augment=False, dataroot=hparams['dataroot'], download=True)

# The data is in the range [-0.5, 0.5]
train_dataloader_cifar = torch.utils.data.DataLoader(train_cifar, batch_size=32, num_workers=0, pin_memory=True)
test_dataloader_cifar = torch.utils.data.DataLoader(test_cifar, batch_size=32, num_workers=0, pin_memory=True)

# The data is in the range [-0.5, 0.5]
train_dataloader_svhn = torch.utils.data.DataLoader(train_svhn, batch_size=32, num_workers=0, pin_memory=True)
test_dataloader_svhn = torch.utils.data.DataLoader(test_svhn, batch_size=32, num_workers=0, pin_memory=True)

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)
model = model.eval()

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')


def save_samples(images, name):
    with open(os.path.join('reconstructions', '{}.pkl'.format(name)), 'wb') as f:
        pkl.dump(images, f)

    images = images + 0.5
    grid = torchvision.utils.make_grid(images[:16], 4, padding=1)  # Should be a 4x4 grid

    fig = plt.figure(figsize=(6,6))
    grid = grid.detach().cpu().numpy().transpose(1,2,0)

    vmax = 1e6
    vmin = -1e6
    large_values = (grid[:,:,0] > vmax) | (grid[:,:,1] > vmax) | (grid[:,:,2] > vmax)
    small_values = (grid[:,:,0] < vmin) | (grid[:,:,1] < vmin) | (grid[:,:,2] < vmin)
    if np.sum(large_values) > 0:
        grid[large_values] = np.stack(np.array([[1,0,0]] * np.sum(large_values)))
    if np.sum(small_values) > 0:
        grid[small_values] = np.stack(np.array([[0,1,1]] * np.sum(small_values)))

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(grid)
    plt.savefig(os.path.join('reconstructions', '{}.png'.format(name)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def compute_total_recons_error(dataloader, name):
    all_recons_errors = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.cuda()
            zn = []
            z, zn, nll, y_logits = model(x=images, zn=zn)

            recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)  # Is temperature 1 correct?
            minibatch_recons_errs = [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, images)]
            all_recons_errors += minibatch_recons_errs

            save_samples(images, '{}_orig_{}'.format(name, i))
            save_samples(recons, '{}_recons_{}'.format(name, i))

    all_recons_errors = np.array(all_recons_errors)
    return all_recons_errors


# minibatch_size = 32
minibatch_size = 25
num_batches = 400

# Gaussian
gaussian_recons_errs = []
for batch_num in range(num_batches):
    print('Gaussian batch {}'.format(batch_num))
    with torch.no_grad():
        # gaussian_data = np.clip(.5 + np.random.normal(size=(minibatch_size, 3, 32, 32)), a_min=-0.49, a_max=0.49)
        gaussian_data = np.clip(np.random.normal(size=(minibatch_size, 3, 32, 32)), a_min=-0.49, a_max=0.49)
        gaussian_data = torch.from_numpy(gaussian_data).float()
        gaussian_data = gaussian_data.cuda()
        zn = []
        z, zn, bpd, y_logits = model(x=gaussian_data, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        gaussian_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, gaussian_data)]
    save_samples(gaussian_data, 'gaussian_orig')
    save_samples(recons, 'gaussian_recons')
print('Gaussian rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(gaussian_recons_errs), np.min(gaussian_recons_errs), np.max(gaussian_recons_errs)))

gaussian_recons_errs = np.array(gaussian_recons_errs)
which_are_inf = np.isinf(gaussian_recons_errs)
which_are_ok = ~which_are_inf
ok_values = gaussian_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('Gaussian ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('Gaussian all inf')
print('Gaussian num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(gaussian_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(gaussian_recons_errs))))
sys.stdout.flush()

# Uniform
uniform_recons_errs = []
for batch_num in range(num_batches):
    print('Uniform batch {}'.format(batch_num))
    with torch.no_grad():
        uniform_data = np.clip(np.random.uniform(size=(minibatch_size, 3, 32, 32)) - 0.5, a_min=-0.49, a_max=0.49)
        uniform_data = torch.from_numpy(uniform_data).float()
        uniform_data = uniform_data.cuda()
        zn = []
        z, zn, nll, y_logits = model(x=uniform_data, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        uniform_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, uniform_data)]
    save_samples(uniform_data, 'uniform_orig')
    save_samples(recons, 'uniform_recons')
print('Uniform rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(uniform_recons_errs), np.min(uniform_recons_errs), np.max(uniform_recons_errs)))

uniform_recons_errs = np.array(uniform_recons_errs)
which_are_inf = np.isinf(uniform_recons_errs)
which_are_ok = ~which_are_inf
ok_values = uniform_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('Uniform ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('Uniform all inf')
print('Uniform num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(uniform_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(uniform_recons_errs))))
sys.stdout.flush()

# Rademacher
rademacher_recons_errs = []
for batch_num in range(num_batches):
    print('Rademacher batch {}'.format(batch_num))
    with torch.no_grad():
        rademacher_data = np.clip(np.random.binomial(1, .5, size=(minibatch_size, 3, 32, 32)) - 0.5, a_min=-0.49, a_max=0.49)
        rademacher_data = torch.from_numpy(rademacher_data).float()
        rademacher_data = rademacher_data.cuda()
        zn = []
        z, zn, nll, y_logits = model(x=rademacher_data, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        rademacher_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, rademacher_data)]
    save_samples(rademacher_data, 'rademacher_orig')
    save_samples(recons, 'rademacher_recons')
print('Rademacher rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(rademacher_recons_errs), np.min(rademacher_recons_errs), np.max(rademacher_recons_errs)))

rademacher_recons_errs = np.array(rademacher_recons_errs)
which_are_inf = np.isinf(rademacher_recons_errs)
which_are_ok = ~which_are_inf
ok_values = rademacher_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('Rademacher ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('Rademacher all inf')
print('Rademacher num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(rademacher_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(rademacher_recons_errs))))
sys.stdout.flush()

# Texture OOD Data
texture_recons_errs = []
for batch_num in range(num_batches):
    print('Texture batch {}'.format(batch_num))
    with torch.no_grad():
        texture_data = np.clip(torch.load(os.path.join('ood_data/dtd.t7')).numpy() / 255.0 - 0.5, a_min=-0.49, a_max=0.49)
        texture_data = torch.from_numpy(texture_data).float()
        texture_data = texture_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size].cuda()
        zn = []
        z, zn, nll, y_logits = model(x=texture_data, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        texture_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, texture_data)]
    save_samples(texture_data, 'texture_orig')
    save_samples(recons, 'texture_recons')
print('Texture rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(texture_recons_errs), np.min(texture_recons_errs), np.max(texture_recons_errs)))

texture_recons_errs = np.array(texture_recons_errs)
which_are_inf = np.isinf(texture_recons_errs)
which_are_ok = ~which_are_inf
ok_values = texture_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('Texture ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('Texture all inf')
print('Texture num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(texture_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(texture_recons_errs))))
sys.stdout.flush()

# Places OOD Data
places_recons_errs = []
for batch_num in range(num_batches):
    print('Places batch {}'.format(batch_num))
    with torch.no_grad():
        places_data = np.clip(torch.load(os.path.join('ood_data/places.t7')).numpy() / 255. - 0.5, a_min=-0.49, a_max=0.49)
        places_data = torch.from_numpy(places_data).float()
        places_data = places_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size].cuda()
        zn = []
        z, zn, nll, y_logits = model(x=places_data, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        places_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, places_data)]
    save_samples(places_data, 'places_orig')
    save_samples(recons, 'places_recons')
print('Places rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(places_recons_errs), np.min(places_recons_errs), np.max(places_recons_errs)))

places_recons_errs = np.array(places_recons_errs)
which_are_inf = np.isinf(places_recons_errs)
which_are_ok = ~which_are_inf
ok_values = places_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('Places ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('Places all inf')
print('Places num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(places_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(places_recons_errs))))
sys.stdout.flush()


transform = torchvision.transforms.Compose([torchvision.transforms.Resize(32),
                                            torchvision.transforms.ToTensor()])
tim_resize_data = torchvision.datasets.ImageFolder('ood_data/Imagenet_resize', transform=transform)
tim_resize_data = np.stack([img.numpy() for (img, label) in tim_resize_data])

tim_recons_errs = []
for batch_num in range(num_batches):
    print('tinyImageNet batch {}'.format(batch_num))
    tim_resize_minibatch = torch.from_numpy(tim_resize_data[batch_num*minibatch_size:batch_num*minibatch_size+minibatch_size]).float() - 0.5  # To get to the range [-0.5, 0.5]
    tim_resize_minibatch = tim_resize_minibatch.cuda()
    with torch.no_grad():
        zn = []
        z, zn, nll, y_logits = model(x=tim_resize_minibatch, zn=zn)
        recons = model.reverse_flow(z.detach(), zn=zn, y_onehot=None, temperature=1, no_grad=True)
        tim_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, tim_resize_minibatch)]
    save_samples(tim_resize_minibatch, 'tim_orig')
    save_samples(recons, 'tim_recons')
print('tinyImageNet rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
       np.mean(tim_recons_errs), np.min(tim_recons_errs), np.max(tim_recons_errs)))

tim_recons_errs = np.array(tim_recons_errs)
which_are_inf = np.isinf(tim_recons_errs)
which_are_ok = ~which_are_inf
ok_values = tim_recons_errs[which_are_ok]
if len(ok_values) > 0:
    print('tinyImageNet ok rec | mean: {:6.4e} | min: {:6.4e} | max: {:6.4e}'.format(
          np.mean(ok_values), np.min(ok_values), np.max(ok_values)))
else:
    print('tinyImageNet all inf')
print('tinyImageNet num inf: {} | num total: {} | proportion inf: {}'.format(
       np.sum(which_are_inf), len(tim_recons_errs), np.sum(which_are_inf).astype(np.float32) / float(len(tim_recons_errs))))
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
