"""
Script containing the invertibility attack for a normalizing flow trained on CelebA64.
"""

import math
import os
import os.path
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from lib.resflow import ResidualFlow
import lib.datasets as datasets
import lib.layers as layers
import lib.utils as utils


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x

def standard_normal_sample(size):
    return torch.randn(size)

def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)

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

def visualize(epoch, model, itr, real_imgs):
    utils.makedirs(os.path.join(args.save, 'imgs'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs

    if args.data == 'celeba64_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    with torch.no_grad():
        # reconstructed real images
        real_imgs, _ = add_padding(real_imgs, nvals)
        if args.squeeze_first: real_imgs = squeeze_layer(real_imgs)
        recon_imgs = model(model(real_imgs.view(-1, *input_size[1:])), inverse=True).view(-1, *input_size[1:])
        if args.squeeze_first: recon_imgs = squeeze_layer.inverse(recon_imgs)
        recon_imgs = remove_padding(recon_imgs)

        # random samples
        fake_imgs = model(fixed_z, inverse=True).view(-1, *input_size[1:])
        if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
        fake_imgs = remove_padding(fake_imgs)

        fake_imgs = fake_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
        recon_imgs = recon_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
        imgs = torch.cat([_real_imgs, fake_imgs, recon_imgs], 0)
        filename = os.path.join(output_folder, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)

device = torch.device("cuda")


###### MODEL SELECTION ######
# Select output folder, where images should be saved
output_folder = 'experiments/affineCelebA64/attacks'
# Select folder
model_name = 'experiments/affineCelebA64/models/checkpt-0087.pth'
# Select if trained model is a Residual Flow (for affine models select False)
resflow = False

# load arguments
checkpoint = torch.load(model_name)
args = checkpoint['args']
args.val_batchsize = 6
args.batchsize = 6


# define data
if args.data == 'celeba64_5bit':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        args.imagesize = 64
    # no dequantization for attack!
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                reduce_bits
            ])
        ), batch_size=args.batchsize, shuffle=False, num_workers=args.nworkers
    )
    # no dequantization for attack!
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebAHQ(train=False, transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        reduce_bits
        ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
    )
else:
    Exception('ERROR: only celebA 64x64 supported')

# define model
if args.task in ['classification', 'hybrid']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification with {}'.format(args.data))
else:
    n_classes = 1

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)
dataset_size = len(train_loader.dataset)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

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

model = model.to(device)
# load model
with torch.no_grad():
    x = torch.rand(1, *input_size[1:]).to(device)
    model(x)
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()

def postprocess(x):
    x = torch.clamp(x, 0.0, 1.0)
    x = x * 255#2**args.nbits
    x += 0.5
    return torch.clamp(x, 0, 255).byte()

# get images
images, labels = iter(train_loader).next()
images = images.to(device)

losses = []

def plot_loss():
    fig = plt.figure()
    plt.plot(range(len(losses)), losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.yscale('log')
    plt.savefig(os.path.join(output_folder, 'attack_loss.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_recons(original_x, x, x_hat, x_hat_2, i, epsilon):
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(postprocess(original_x[0]).detach().cpu().numpy().transpose(1,2,0))
    ax[1].imshow(postprocess(x[0]).detach().cpu().numpy().transpose(1,2,0))
    ax[2].imshow(postprocess(x_hat[0]).detach().cpu().numpy().transpose(1,2,0))
    ax[3].imshow(postprocess(x_hat_2[0]).detach().cpu().numpy().transpose(1,2,0))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'attack_{}_{}.png'.format(epsilon, i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)


###### HYPERPARAMETER SETTING ######
epsilon = 0.5* (1. / 2**args.nbits)  # size of maximal perturbation
k = 200  # number of attack iterations
a = 0.0005 # step size
random_start = True

# select which image to attack
original_x = images[0].view(-1, im_dim, args.imagesize, args.imagesize)

recon_errors = []
recon_errors_loss = []


def reconstruct(model, img, num_iter=None):
    if args.data == 'celeba64_5bit':
        nvals = 32
    elif args.data == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256
    # reconstructed real images
    img, _ = add_padding(img, nvals)
    if args.squeeze_first: img = squeeze_layer(img)
    z = model(img.view(-1, *input_size[1:]), logpx=None)
    if num_iter is not None:
        recon_imgs = model.inverse(z, logpz=None, num_iter=num_iter).view(-1, *input_size[1:])
    else:
        recon_imgs = model.inverse(z, logpz=None).view(-1, *input_size[1:])
    if args.squeeze_first: recon_imgs = squeeze_layer.inverse(recon_imgs)
    recon_imgs = remove_padding(recon_imgs)
    recon_imgs = recon_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
    return recon_imgs


def perturb(X_nat):
    '''
    Given examples (X_nat, y), returns non-invertible adversarial example
    within epsilon of X_nat in l_infinity norm.
    '''
    # center data in dequantization cube (no dequantization performed on orig data)
    nvals = 2**args.nbits
    noise = X_nat.new().resize_as_(X_nat).uniform_()
    X_nat = X_nat * (nvals - 1) + noise / 2.
    X_nat = X_nat / nvals

    if random_start:
        X = X_nat + (2 * epsilon) * torch.rand(X_nat.shape).cuda() - epsilon
        X = torch.clamp(X, X_nat.min(), X_nat.max())

    for i in range(k):
        X = X.clone().detach().requires_grad_(True)

        if resflow:
            x_hat = reconstruct(model, X, num_iter=4)
        else:
            x_hat = reconstruct(model, X)

        loss = 1 - torch.norm(X - x_hat)
        losses.append(loss.item())

        # compute reconstruction error for logging
        recon_error_loss = torch.norm(X - x_hat).item()
        recon_errors_loss.append(recon_error_loss)

        # compute reconstruction error with more iterations (for resflow)
        # (for affine models, its the same computations)
        with torch.no_grad():
            if resflow:
                x_hat_2 = reconstruct(model, X, num_iter=200)
            else:
                x_hat_2 = reconstruct(model, X)
            recon_error = torch.norm(X - x_hat_2).item()
        recon_errors.append(recon_error)

        # gradient update step
        X.grad = torch.autograd.grad(loss, X)[0]
        with torch.no_grad():
            X.data -= a * torch.sign(X.grad.data)
            #X.data -= a * X.grad.data
            X = torch.max(torch.min(X, X_nat + epsilon), X_nat - epsilon)
            X = torch.clamp(X, X_nat.min(), X_nat.max()) # ensure valid pixel range

        print('Iter: {:6.4e}|Loss: {:6.4e} |x-xhat: {:6.4e}|x-xhat2: {:6.4e} | x-xorig: {:6.4e}'.format(
               i, loss, recon_error_loss, recon_error, torch.norm(X - X_nat).item()))

        plot_loss()
        plot_recons(X_nat, X, x_hat, x_hat_2, i, epsilon)

        # save recon error numbers as numpy array
        rec_npy = np.array(recon_errors)
        np.save(os.path.join(output_folder, 'recon_errors'), rec_npy)
        rec_loss_npy = np.array(recon_errors_loss)
        np.save(os.path.join(output_folder, 'recon_errors_loss'), rec_loss_npy)

    return X

perturb(original_x)
