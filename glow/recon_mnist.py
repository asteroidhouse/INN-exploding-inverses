import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_CIFAR10, get_SVHN, get_MNIST, postprocess
from model import Glow
import ipdb
import os
import argparse
import copy
from torchvision import utils as torchvision_utils
import numpy as np

import torch.utils.data as data
device = torch.device("cuda")


def plot_imgs(imgs, title):
    K = len(imgs)
    f, axs = plt.subplots(1,K,figsize=(K*5,4))
    for idx, images in enumerate(imgs):
        grid = make_grid((postprocess(images).cpu().detach()[:30]), nrow=6).permute(1,2,0)
        axs[idx].imshow(grid)
        axs[idx].axis('off')
    plt.savefig(os.path.join(output_folder, f'{title}.png'))


def abs_pixel_dist(x1, x2):
    def _quant(x):
        return (x+.5) * 255
    return torch.abs(_quant(x1) - _quant(x2)).mean().item()

def run_recon(x, model):
    z = model(x, None, correction=False)[0]
    recon = model(y_onehot=None, temperature=1, z=z, reverse=True, use_last_split=True)
    return recon

def run_recon_evolution(model, x, fpath, return_dict=False):
    errs = []
    l2errs = []
    oimg = copy.deepcopy(x)
    xs = [x]
    recon_errs = []
    for _ in range(9):
        recon = run_recon(x, model).detach()
        errs.append(abs_pixel_dist(oimg, recon))
        l2errs.append(torch.mean((oimg-recon)**2).item())
        xs.append(recon)
        recon_errs.append(recon - oimg)
        x = recon

    fig, axs = plt.subplots(1, 4, figsize=(20,4))
    grid = .5+torchvision_utils.make_grid(torch.cat([x[:10] for x in xs],0)[:100], nrow=10, padding=1, pad_value=1.0)
    axs[0].imshow(grid.permute(1,2,0).detach().cpu())
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Recons")
    grid = .5+torchvision_utils.make_grid(torch.cat([x[:10] for x in recon_errs],0)[:100], nrow=10, padding=1, pad_value=1.0)
    axs[1].imshow(grid.permute(1,2,0).detach().cpu())
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("Erros")
    axs[2].plot(np.arange(len(errs)),  errs)
    axs[2].set_ylabel('Abs Pixel Dist')
    axs[2].set_xlabel('Recon Pass')
    axs[3].plot(np.arange(len(errs)),  l2errs)
    axs[3].set_ylabel('L2 Dist per pixel')
    axs[3].set_xlabel('Recon Pass')
    axs[0].axis('off')
    axs[1].axis('off')
    fig.suptitle('Each row is every 5 foward/inverse pass', fontsize=16)
    plt.savefig(fpath)
    # return the final PAD
    if return_dict:
        ret = {
            'errs':errs,
            'l2errs':l2errs
        }
    else:
        ret = errs[-1]
    return ret

def main(args,kwargs):
    output_folder = args.output_folder
    model_name = args.model_name

    with open(os.path.join(output_folder,'hparams.json')) as json_file:
        hparams = json.load(json_file)

    image_shape, num_classes, _, test_mnist = get_MNIST(False, hparams['dataroot'], hparams['download'])
    test_loader = data.DataLoader(test_mnist, batch_size=32,
                                      shuffle=False, num_workers=6,
                                      drop_last=False)
    x, y = test_loader.__iter__().__next__()
    x = x.to(device)

    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'], False if 'logittransform' not in hparams else hparams['logittransform'],False if 'sn' not in hparams else hparams['sn'])

    model.load_state_dict(torch.load(os.path.join(output_folder, model_name)))
    model.set_actnorm_init()

    model = model.to(device)
    model = model.eval()


    with torch.no_grad():
        images = model(y_onehot=None, temperature=1, batch_size=32, reverse=True).cpu()
        better_dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True, use_last_split=True).cpu()
        dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True).cpu()
        worse_dup_images = model(y_onehot=None, temperature=1, z=model._last_z, reverse=True).cpu()

    l2_err =  torch.pow((images - dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
    better_l2_err =  torch.pow((images - better_dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
    worse_l2_err =  torch.pow((images - worse_dup_images).view(images.shape[0], -1), 2).sum(-1).mean()
    print(l2_err, better_l2_err, worse_l2_err)
    plot_imgs([images, dup_images, better_dup_images, worse_dup_images], '_recons')

    with torch.no_grad():
        z, nll, y_logits = model(x, None)
        better_dup_images = model(y_onehot=None, temperature=1, z=z, reverse=True, use_last_split=True).cpu()

    plot_imgs([x, better_dup_images], '_data_recons2')

    fpath = os.path.join(output_folder, '_recon_evoluation.png')
    pad = run_recon_evolution(model, x, fpath)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_folder', type=str,
                        default= '.')

    parser.add_argument('--model_name',
                        type=str, default='glow_model_250.pth')
    args = parser.parse_args()
    kwargs = vars(args)

    main(args, kwargs)
