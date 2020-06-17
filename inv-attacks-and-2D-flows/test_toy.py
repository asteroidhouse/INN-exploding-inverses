import matplotlib.pyplot as plt

import pdb

import argparse
import os
import time
import math
import numpy as np

import torch

import lib.optimizers as optim
import lib.layers.base as base_layers
import lib.layers as layers
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

LOW = -4
HIGH = 4


parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str,
                    default='experiments/iresnet_toyL2_07_100blocks')
args_new = parser.parse_args()

def parse_vnorms():
    ps = []
    for p in args.vnorms:
        if p == 'f':
            ps.append(float('inf'))
        else:
            ps.append(float(p))
    return ps[:-1], ps[1:]

def build_nnet(dims, activation_fn=torch.nn.ReLU):
    nnet = []
    domains, codomains = parse_vnorms()
    if args.learn_p:
        if args.mixed:
            domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
        else:
            domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
        codomains = domains[1:] + [domains[0]]
    for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
        nnet.append(activation_fn())
        nnet.append(
            base_layers.get_linear(
                in_dim,
                out_dim,
                coeff=args.coeff,
                n_iterations=args.n_lipschitz_iters,
                atol=args.atol,
                rtol=args.rtol,
                domain=domain,
                codomain=codomain,
                zero_init=(out_dim == 2),
            )
        )
    return torch.nn.Sequential(*nnet)

def build_nnet_affine(dims, activation_fn=torch.nn.ReLU):
    nnet = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        nnet.append(activation_fn())
        if i == 0:
            nnet.append(torch.nn.Linear(in_dim // 2, out_dim))
        else:
            nnet.append(torch.nn.Linear(in_dim, out_dim))
    return torch.nn.Sequential(*nnet)



# load model
device = 'cuda'
model_path = args_new.save + '/checkpt.pth'
checkpoint = torch.load(model_path, map_location=device)
args = checkpoint['args']
ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
}
activation_fn = ACTIVATION_FNS[args.act]
dims = [2] + list(map(int, args.dims.split('-'))) + [2]
if args.arch == 'iresnet':
    blocks = []
    if args.actnorm: blocks.append(layers.ActNorm1d(2))
    for _ in range(args.nblocks):
        blocks.append(
            layers.iResBlock(
                build_nnet(dims, activation_fn),
                n_dist=args.n_dist,
                n_power_series=args.n_power_series,
                exact_trace=args.exact_trace,
                brute_force=args.brute_force,
                n_samples=args.n_samples,
                neumann_grad=False,
                grad_in_forward=False,
            )
        )
        if args.actnorm: blocks.append(layers.ActNorm1d(2))
        if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
    model = layers.SequentialFlow(blocks).to(device)
elif args.arch == 'realnvp':
    blocks = []
    for _ in range(args.nblocks):
        blocks.append(layers.CouplingBlock(2, build_nnet_affine(dims, activation_fn), swap=False))
        blocks.append(layers.CouplingBlock(2, build_nnet_affine(dims, activation_fn), swap=True))
        if args.actnorm: blocks.append(layers.ActNorm1d(2))
        if args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
    model = layers.SequentialFlow(blocks).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

# checkerboard data
def checkerboard(batch_size):
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


def det_2x2(x):
    return x[0,0]*x[1,1]-x[0,1]*x[1,0]

def compute_det(inputs, outputs):
    batch_size = outputs.size(0)
    outVector = torch.sum(outputs,0).view(-1)
    outdim = outVector.size()[0]
    jac = torch.stack([torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].view(batch_size, outdim) for i in range(outdim)], dim=1)
    det = torch.stack([det_2x2(jac[i,:,:]) for i in range(batch_size)], dim=0)
    return det, jac

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2



def plot_griddata(xx, yy, grid_data, min_value, max_value, name, change_min=None):
    grid_data = grid_data.reshape(xx.shape[0], yy.shape[0])
    grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)
    grid_data = np.ma.masked_where(~np.isfinite(grid_data), grid_data)
    if grid_data.min() == min_value:
        if change_min is None:
            min_plot = 1e-8
            grid_data[grid_data <= min_value] = 1e-8
        else:
            min_plot = change_min
            grid_data[grid_data <= min_value] = change_min
    else:
        min_plot = grid_data.min()
    if grid_data.max() > max_value:
        max_plot = max_value
    else:
        max_plot = grid_data.max()
    plt.figure()
    import matplotlib.colors as colors
    plt.pcolor(xx, yy, grid_data,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
    plt.colorbar()
    plt.savefig(args_new.save + name)

# create grid
domain = 8
resolution = 0.05
x = np.arange(-domain, domain, resolution)
y = np.arange(-domain, domain, resolution)
xx, yy = np.meshgrid(x, y, sparse=False)
list_grid = np.array([xx, yy]).reshape(2, xx.shape[0]*yy.shape[0]).transpose()
grid_path = str(domain) + '_' + str(resolution)[2:]

#################################
# reconstruction error on grid
batch_size = 10000
num_batches = int(np.ceil(list_grid.shape[0] / batch_size))
recon_error_grid = torch.tensor([]).to(device)
recon_error_out_grid = torch.tensor([]).to(device)
for i in range(num_batches):
    print(str(i) + ' of ' + str(num_batches - 1))
    if (i+1)* batch_size <= list_grid.shape[0]:
        input_data = list_grid[i*batch_size: (i+1)*batch_size]
    else:
        input_data = list_grid[i*batch_size: ]
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
    input_data.requires_grad = False
    zero = torch.zeros(input_data.shape[0], 1).to(input_data)
    # recon error for samples from input grid
    with torch.no_grad():       
        out_forw, _  = model.forward(input_data, zero)
        x_recon = model.inverse(out_forw)
        recon_error_grid = torch.cat((recon_error_grid, torch.norm(input_data - x_recon, dim=1)), 0)
    # recon error for samples from output grid
    with torch.no_grad():
        out_inv = model.inverse(input_data)
        z_recon, _ = model.forward(out_inv, zero)
        recon_error_out_grid = torch.cat((recon_error_out_grid, torch.norm(input_data - z_recon, dim=1)), 0)
recon_error_grid = recon_error_grid.detach().cpu().numpy()
recon_error_out_grid = recon_error_out_grid.detach().cpu().numpy()

plot_griddata(xx, yy, recon_error_grid, 0.0, 1, '/recon-error-area' + grid_path)
plot_griddata(xx, yy, recon_error_out_grid, 0.0, 1, '/recon-error-output-area' + grid_path)
save_path_recon = args_new.save + '/recon-error-area' + grid_path
save_path_recon_out = args_new.save + '/recon-error-output-area' + grid_path
np.save(save_path_recon, recon_error_grid)
np.save(save_path_recon_out, recon_error_out_grid)
print('recon error grid plotting done')
del recon_error_grid, x_recon
del recon_error_out_grid, z_recon

#################################
# log det error and condition number on grid
batch_size = 1000
num_batches = int(np.ceil(list_grid.shape[0] / batch_size))
logdet_error_grid = np.array([])
logdet_error_grid_w = np.array([])
cond_error_grid = np.array([])
likelihood_grid = np.array([])
pz_grid = np.array([])
logdet_grid = np.array([])
for i in range(num_batches):
    print(str(i) + ' of ' + str(num_batches - 1))
    if (i+1)* batch_size <= list_grid.shape[0]:
        input_data = list_grid[i*batch_size: (i+1)*batch_size]
    else:
        input_data = list_grid[i*batch_size: ]
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
    input_data.requires_grad = True
    zero = torch.zeros(input_data.shape[0], 1).to(input_data)
    out_forw, logdet_ana  = model.forward(input_data, zero)
    det_numeric, jac = compute_det(input_data, out_forw)
    logdet_numeric = - torch.log(det_numeric)
    error_logdet = torch.norm(logdet_numeric.view(logdet_numeric.shape[0], 1)  - logdet_ana, dim=1)
    error_logdet = error_logdet.detach().cpu().numpy()
    logdet_error_grid = np.concatenate((logdet_error_grid, error_logdet), axis=0)
    # compute likelihood
    logpz = standard_normal_logprob(out_forw).sum(1, keepdim=False)
    logpx = logpz - logdet_ana.view(-1)
    pz = np.exp((logpz).detach().cpu().numpy())
    logdet = np.exp((-logdet_ana.view(-1)).detach().cpu().numpy())
    likelihood = np.exp((logpx).detach().cpu().numpy())
    pz_grid = np.concatenate((pz_grid, pz), axis=0)
    logdet_grid = np.concatenate((logdet_grid, logdet), axis=0)
    likelihood_grid = np.concatenate((likelihood_grid, likelihood), axis=0)    
    # divide error by likelihood
    error_weighted = error_logdet / np.abs(logpx.detach().cpu().numpy())
    logdet_error_grid_w = np.concatenate((logdet_error_grid_w, error_weighted), axis=0)
    # compute condition number
    jac = jac.detach().cpu().numpy()
    cond_num = np.linalg.cond(jac)
    cond_error_grid = np.concatenate((cond_error_grid, cond_num), axis=0)
    
    
# plotting of log det error
plot_griddata(xx, yy, logdet_error_grid, 0.0, 10, '/logdet-error-area'+ grid_path)
# plotting of weighted log det error
plot_griddata(xx, yy, logdet_error_grid_w, 0.0, 10, '/logdet-error-weighted-area'+ grid_path)
# plotting of condition number
plot_griddata(xx, yy, cond_error_grid, 1.0, 1e10, '/condition_num-area'+ grid_path, change_min=1.0)
# plot likelihood
plot_griddata(xx, yy, likelihood_grid, 0.0, 1e5, '/likelihood-area'+ grid_path)
# plot pz
plot_griddata(xx, yy, pz_grid, 0.0, 1e5, '/pz-area'+ grid_path)
# plot logdet
plot_griddata(xx, yy, logdet_grid, 0.0, 1e5, '/det-area'+ grid_path)
# save data
save_path_logdet = args_new.save + '/logdet-error-area' + grid_path
save_path_logdet_weighted = args_new.save + '/logdet-error-weighted-area' + grid_path
save_path_cond = args_new.save + '/condition_num-area' + grid_path
save_path_like = args_new.save + '/likelihood-area' + grid_path
save_path_pz = args_new.save + '/pz-area' + grid_path
save_path_det = args_new.save + '/det-area' + grid_path
np.save(save_path_logdet, logdet_error_grid)
np.save(save_path_logdet_weighted, logdet_error_grid_w)
np.save(save_path_cond, cond_error_grid)
np.save(save_path_like, likelihood_grid)
np.save(save_path_pz, pz_grid)
np.save(save_path_det, logdet_grid)
