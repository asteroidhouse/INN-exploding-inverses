"""Train a Glow model on a toy 2D regression problem.

Example
-------
python toy_2d_regression.py --nf_coeff=0
python toy_2d_regression.py --nf_coeff=1e-6
"""
import sys
import ipdb
import argparse

import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.gradcheck import zero_gradients

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument('--nf_coeff', type=float, default=0.0,
                    help='Coefficient for the normalizing flow regularizer')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1111)
torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)


def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps

def standard_gaussian(shape):
    mean, logsd = [torch.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)

def gaussian_diag(mean, logsd):
    class o(object):
        Log2PI = float(np.log(2 * np.pi))
        pass

        def logps(x):
            return  -0.5 * (o.Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        def sample():
            eps = torch.zeros_like(mean).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp = lambda x: flatten_sum(o.logps(x))
    return o


class LinearZeroInit(nn.Linear):
    def reset_parameters(self):
        self.weight.data.fill_(0.)
        self.bias.data.fill_(0.)


def NN(in_channels, hidden_channels=128, channels_out=None):
    channels_out = channels_out or in_channels

    return nn.Sequential(nn.Linear(in_channels, hidden_channels),
                         nn.ReLU(),
                         nn.Linear(hidden_channels, hidden_channels),
                         nn.ReLU(),
                         nn.Linear(hidden_channels, channels_out))


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError


class LayerList(Layer):
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward_(self, x, objective):
        for layer in self.layers:
            x, objective = layer.forward_(x, objective)
        return x, objective

    def reverse_(self, x, objective):
        for layer in reversed(self.layers):
            x, objective = layer.reverse_(x, objective)
        return x, objective


class Shuffle(Layer):
    def __init__(self, num_channels):
        super(Shuffle, self).__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(num_channels):
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer('indices', indices)
        self.register_buffer('rev_indices', rev_indices)

    def forward_(self, x, objective):
        return x[:, self.indices], objective

    def reverse_(self, x, objective):
        return x[:, self.rev_indices], objective


class Reverse(Shuffle):
    def __init__(self, num_channels):
        super(Reverse, self).__init__(num_channels)
        indices = np.copy(np.arange(num_channels)[::-1])
        indices = torch.from_numpy(indices).long()
        self.indices.copy_(indices)
        self.rev_indices.copy_(indices)


class AdditiveCoupling(Layer):
    def __init__(self, num_features):
        super(AdditiveCoupling, self).__init__()
        self.NN = NN(num_features // 2, hidden_channels=10, channels_out=num_features // 2)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 = z2 + self.NN(z1)
        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 = z2 - self.NN(z1)
        return torch.cat([z1, z2], dim=1), objective


class AffineCoupling(Layer):
    def __init__(self, num_features):
        super(AffineCoupling, self).__init__()
        self.affine_scale_fn = 'sigmoid'
        self.NN = NN(num_features // 2, hidden_channels=10, channels_out=num_features)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]

        if self.affine_scale_fn == 'sigmoid':
            scale = F.sigmoid(h[:, 1::2] + 2.)
        elif self.affine_scale_fn == 'exp':
            scale = torch.exp(h[:, 1::2])

        z2 = z2 + shift
        z2 = z2 * (scale + 1e-6)
        objective = objective + flatten_sum(torch.log(scale))

        return torch.cat([z1, z2], dim=1), objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]

        if self.affine_scale_fn == 'sigmoid':
            scale = F.sigmoid(h[:, 1::2] + 2.)
        elif self.affine_scale_fn == 'exp':
            scale = torch.exp(h[:, 1::2])

        z2 = z2 / (scale + 1e-6)
        z2 = z2 - shift
        objective = objective - flatten_sum(torch.log(scale))
        return torch.cat([z1, z2], dim=1), objective


class ActNorm(Layer):
    def __init__(self, num_features, logscale_factor=1., scale=1., stable_eps=0):
        super(Layer, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features)))
        self.stable_eps = stable_eps

    def forward_(self, input, objective):
        input_shape = input.size()
        logs = self.logs * self.logscale_factor
        b = self.b
        output = (input + b) * (torch.exp(logs) +  self.stable_eps)
        dlogdet = torch.sum(torch.log(torch.exp(self.logs * self.logscale_factor) +  self.stable_eps)) * input.size(-1) # c x h
        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective):
        # assert self.initialized

        input_shape = input.size()
        # input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input / (torch.exp(logs) + self.stable_eps) - b
        dlogdet = torch.sum(torch.log(torch.exp(logs) +  self.stable_eps)) * input.size(-1) # c x h

        return output.view(input_shape), objective - dlogdet


class RevNetStep(LayerList):
    def __init__(self, num_features, coupling='additive', permutation='shuffle', use_actnorm=False):
        super(RevNetStep, self).__init__()

        layers = []

        if use_actnorm:
            layers += [ActNorm(num_features)]

        if permutation == 'reverse':
            layers += [Reverse(num_features)]
        elif permutation == 'shuffle':
            layers += [Shuffle(num_features)]

        if coupling == 'additive':
            layers += [AdditiveCoupling(num_features)]
        elif coupling == 'affine':
            layers += [AffineCoupling(num_features)]

        self.layers = nn.ModuleList(layers)


class GaussianPrior(Layer):
    def __init__(self, input_shape):
        super(GaussianPrior, self).__init__()
        self.input_shape = input_shape

    def forward_(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        pz = gaussian_diag(mean, logsd)
        objective += pz.logp(x)

        # this way, you can encode and decode back the same image.
        return x, objective

    def reverse_(self, x, objective):
        z = x
        # this way, you can encode and decode back the same image.
        return z, objective


class Glow(LayerList, nn.Module):
    def __init__(self, input_shape, coupling='additive', permutation='reverse', num_features=2, depth=2, n_levels=1, use_actnorm=False):
        super(Glow, self).__init__()
        layers = []

        for i in range(n_levels):
            layers += [RevNetStep(num_features, coupling=coupling, permutation=permutation, use_actnorm=use_actnorm) for _ in range(depth)]

        layers += [GaussianPrior(input_shape)]

        self.layers = nn.ModuleList(layers)
        self.flatten()

    def forward(self, *inputs):
        return self.forward_(*inputs)

    def sample(self, x, no_grad=True):
        if no_grad:
            with torch.no_grad():
                samples = self.reverse_(x, 0.)[0]
                return samples
        else:
            samples = self.reverse_(x, 0.)[0]
            return samples

    def flatten(self):
        # flattens the list of layers to avoid recursive call every time.
        processed_layers = []
        to_be_processed = [self]
        while len(to_be_processed) > 0:
            current = to_be_processed.pop(0)
            if isinstance(current, LayerList):
                to_be_processed = [x for x in current.layers] + to_be_processed
            elif isinstance(current, Layer):
                processed_layers += [current]

        self.layers = nn.ModuleList(processed_layers)


def computeSVDjacobian(input, model, inverse=False, linear=False):
    model.eval()

    dims = input[0].size()
    sample = input[0].view(1, dims[0])  # makes the batch size = 1
    sample.requires_grad = True

    if linear:
        z = model(sample)
    else:
        z, _ = model(sample, torch.zeros_like(sample))

    # compute jacobian (for single sample)
    if not inverse:
        out_vector = z.view(-1)
        outdim = out_vector.size(0)
        indim = sample.view(-1).size(0)
        jac = torch.zeros([outdim, indim])
        for i in range(outdim):
            zero_gradients(sample)
            # if i % 100 == 0:
            #     print(i)
            jac[i,:] = grad(out_vector[i], sample, retain_graph=True)[0].view(-1)
        jac = jac.numpy()

        # compute SVD of jacobian
        Ujac, Djac, Vjac = np.linalg.svd(jac, compute_uv=True, full_matrices=False)
    else:
        # invert bijective output
        try:
            x_inv = model.module.sample(z, no_grad=False)
        except:
            x_inv = model.module.inverse(z, no_grad=False)

        out_vector = x_inv.view(-1)
        outdim = out_vector.size(0)
        indim = z.view(-1).size(0)
        jac = torch.zeros([outdim, indim])

        for i in range(outdim):
            zero_gradients(sample)
            if i % 100 == 0:
                print(i)
            jac[i,:] = torch.autograd.grad(out_vector[i], z, retain_graph=True)[0].view(-1)
        jac = jac.numpy()

        # compute SVD of jacobian
        Ujac, Djac, Vjac = np.linalg.svd(jac, compute_uv=True, full_matrices=False)

    model.train()  # Make sure we undo the eval() call at the start of the function
    return Ujac, Djac, Vjac


# Data generation
# ---------------
xy_dim1 = np.random.multivariate_normal([0, 0], [[1,0], [0,1e-24]], size=(10000,))
xy_dim2 = np.random.multivariate_normal([0, 0], [[1,1], [1,1]], size=(10000,))

x_np = np.stack([xy_dim1[:,0], xy_dim2[:,0]]).T
y_np = np.stack([xy_dim1[:,1], xy_dim2[:,1]]).T

x = torch.from_numpy(x_np).float().cuda()
y = torch.from_numpy(y_np).float().cuda()
# ---------------


save_fname = 'glow_nf_{}'.format(args.nf_coeff)

model = Glow(input_shape=x.shape, coupling='affine', permutation='reverse', depth=8, n_levels=3, use_actnorm=True)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

iterations = []
loss_list = []
min_sv_list = []
max_sv_list = []
cond_num_list = []
recons_errors = []

lossFct = torch.nn.MSELoss()

for i in range(40000):
    z, obj = model(x, 0)
    loss = lossFct(z, y) + args.nf_coeff * (-obj.mean())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        z, _ = model(x, 0)
        recons = model.sample(z, no_grad=True)

        recons_err = torch.norm(recons - x, dim=1)
        recons_errors.append(torch.mean(recons_err))
        highest_recons_err_idx = torch.argsort(recons_err, descending=True)[0]

        # Compute condition number, max and min singular values
        U, D, V = computeSVDjacobian(x[highest_recons_err_idx].unsqueeze(0), model, inverse=False, linear=False)
        condition_num = float(D.max() / D.min())
        min_sv = float(D.min())
        max_sv = float(D.max())

        iterations.append(i)
        loss_list.append(loss.item())
        min_sv_list.append(min_sv)
        max_sv_list.append(max_sv)
        cond_num_list.append(condition_num)

        # torch.save(model, '{}.pt'.format(save_fname))

        print('Iter {:5d} | Loss: {:6.4e} | Recons: {:6.4e} | Cond: {:6.4e} | Min SV: {:6.4e} | Max SV: {:6.4e}'.format(
               i, loss.item(), torch.mean(recons_err), condition_num, min_sv, max_sv))
        sys.stdout.flush()

with open('{}.pkl'.format(save_fname), 'wb') as f:
    pkl.dump({ 'iterations': iterations,
               'recons_errors': recons_errors,
               'loss_list': loss_list,
               'min_sv_list': min_sv_list,
               'max_sv_list': max_sv_list,
               'cond_num_list': cond_num_list }, f)
