"""Adapted from https://github.com/jhjacobsen/fully-invertible-revnet
"""
import os
import ipdb
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from torch.autograd.gradcheck import zero_gradients


def computeSVDjacobian(input, model, inverse=False, model_type='irevnet'):
    model.eval()

    dims = input[0].size()
    sample = input[0].view(1, dims[0], dims[1], dims[2])  # makes the batch size = 1
    sample.requires_grad = True

    if model_type == 'iresnet':
        out, z = model(sample)
    elif model_type == 'glow':
        zn = []
        z, zn, nll, y_logits = model(x=sample, zn=zn, y_onehot=torch.zeros((1, 10), device=sample.device))  # Does the y one_hot matter here??
    else:
        z, _ = model(sample, torch.zeros_like(sample[:, 0, 0, 0]))

    # compute jacobian (for single sample)
    if not inverse:
        if model_type == 'glow':
            out_vector = torch.cat([z.view(-1)] + [z_part.view(-1) for z_part in zn])
        else:
            out_vector = z.view(-1)

        outdim = out_vector.size(0)
        indim = sample.view(-1).size(0)
        jac = torch.zeros([outdim, indim])
        for i in range(outdim):
            zero_gradients(sample)
            if i % 100 == 0:
                print(i)
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


def angle_between(v, w):
    """Computes the angle (in radians) between two vectors v and w (of arbitrary dimension).
    """
    unit_v = F.normalize(v, dim=0)  # Get unit vector v
    unit_w = F.normalize(w, dim=0)  # Get unit vector w
    return unit_v.dot(unit_w).clamp(-1.0 + 1e-9, 1.0 - 1e-9).acos()


def flatten(list_of_lists):
    return [item for lst in list_of_lists for item in lst]


def one_hot(labels, num_classes):
    labels_onehot = torch.FloatTensor(labels.size(0), num_classes).to(labels.device)
    labels_onehot.zero_()
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_onehot


# For dataloading with only one class
def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2**n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


# ------------------------------------------------------------------------------
# Utility Methods
# ------------------------------------------------------------------------------
def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps


# ------------------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------------------
def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
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
