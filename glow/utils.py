import math
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad
import numpy as np
import ipdb

def computeSVDjacobian(x, model, compute_inverse=True):
    ret_dict={}
    x = x[:1]
    batch_size = x.size(0)

    model.eval()
    x.requires_grad_()
    z = model.forward(x, None, correction=False, return_details=True)[0]
    old_z = z.clone()
    zs = [split._last_z2.view(x.size(0),-1) for  split  in model.flow.splits]
    zs = zs + [z.view(batch_size, -1)]
    dim = x.view(batch_size, -1).size(1)
    jac = np.zeros([dim, dim])
    dim_counter = 0
    for z in zs:
        for dim in range(z.size(1)):
            zero_gradients(x)
            if dim_counter % 100 == 0: print(dim_counter)
            g = grad(z[:, dim].sum(), x, retain_graph=True)[0].detach()
            jac[dim_counter,:] = g.mean(0).view(-1).cpu().numpy()
            dim_counter+=1
    try:
        Djac = np.linalg.svd(jac, compute_uv=True, full_matrices=False)[1]
    except np.linalg.LinAlgError:
        Djac = np.zeros(len(jac))

    ret_dict['D_for'] = Djac
    ret_dict['jac_for'] = jac
    # inverse
    if compute_inverse:
        old_z.requires_grad_()
        recon = model(y_onehot=None, temperature=1, z=old_z, reverse=True, use_last_split=True) 
        zs = [split._last_z2 for  split  in model.flow.splits]
        zs = zs + [old_z]
        dim = x.view(batch_size, -1).size(1)
        jac = torch.zeros([dim, dim])
        recon_flat = recon.view(batch_size, -1)
        for dim_counter in range(recon_flat.size(1)):
            if dim_counter % 100 == 0: print(dim_counter)
            gs = []
            for z in zs:
                zero_gradients(z)
                g = grad(recon_flat[:, dim_counter].sum(), z, retain_graph=True)[0].detach()
                gs.append(g.view(batch_size,-1))
            jac[dim_counter,:] = torch.cat(gs,-1).mean(0).view(-1)
            dim_counter+=1
        jac = jac.numpy()
        U_inv, D_inv, V_inv = np.linalg.svd(jac, compute_uv=True, full_matrices=False)
        ret_dict['D_inv'] = D_inv
        ret_dict['jac_inv'] = jac
    model.train()
    return ret_dict



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
    x = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

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
