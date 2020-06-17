"""Adapted from https://github.com/jhjacobsen/fully-invertible-revnet
"""
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from utils import *
from layers import *


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]

# ------------------------------------------------------------------------------
# Abstract Classes to define common interface for invertible functions
# ------------------------------------------------------------------------------

# Abstract Class for bijective functions
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective):
        raise NotImplementedError

    def reverse_(self, y, objective):
        raise NotImplementedError


# Wrapper for stacking multiple layers
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

    def gradient_(self, out, grad_output):
        gradients = []
        for layer in reversed(self.layers):
            out, grad_output, layergrad = layer.gradient_(out, grad_output)
            gradients += layergrad
        return grad_output, reversed(gradients)  # So that the gradients are in the same order as model.parameters()

    def gradient_forward_(self, out, grad_output, out_ForwPass, noise_dir=None, noise_eps=None):
        gradients_inverse = []
        for layer in self.layers:
            out, grad_output, layergrad = layer.gradient_forward_(out, grad_output)
            gradients_inverse += layergrad
        gradients_forward = []
        # re-use output of forward pass (from computation of loss) here to
        # reduce the numerical errors during inverse pass
        # (inverse pass is done during backprop to recomputate activations)
        out = out_ForwPass
        out_test = out - noise_eps * noise_dir # remove influence of noise
        recon_error = (torch.norm(out_ForwPass - out_test) / out_ForwPass.shape[0]).item()
        print('Recon error in inverse: ' + str(recon_error))
        for layers in reversed(self.layers):
            out, grad_output, layergrad = layers.gradient_(out, grad_output)
            gradients_forward += layergrad
        gradients_forward = [g for g in gradients_forward if g is not None]
        gradients_inverse = [g for g in gradients_inverse if g is not None]
        gradients = [g_inv + g_forw for (g_inv, g_forw) in zip(gradients_inverse, reversed(gradients_forward))]
        return grad_output, gradients, out


# ------------------------------------------------------------------------------
# Permutation Layers
# ------------------------------------------------------------------------------

# Shuffling on the channel axis
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

    def gradient_(self, out, grad_output):
        with torch.no_grad():
            x = out[:, self.rev_indices]

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x.requires_grad = True

            y = x[:, self.indices]

            # perform full backward pass on graph
            grad_input = torch.autograd.grad(y, x, grad_output)[0]

            # cleanup sub-graph
            y.detach_()
            del y

        return x, grad_input, [None]  # None because there are no parameters in this class

    def gradient_forward_(self, out, grad_output):
        with torch.no_grad():
            y = out[:,  self.indices]

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y.requires_grad = True

            x = y[:, self.rev_indices]

            # perform full backward pass on graph
            grad_input = torch.autograd.grad(x, y, grad_output)[0]

            # cleanup sub-graph
            x.detach_()
            del x

        return y, grad_input, [None]  # None because there are no parameters in this class


# Reversing on the channel axis
class Reverse(Shuffle):
    def __init__(self, num_channels):
        super(Reverse, self).__init__(num_channels)
        indices = np.copy(np.arange(num_channels)[::-1])
        indices = torch.from_numpy(indices).long()
        self.indices.copy_(indices)
        self.rev_indices.copy_(indices)


# Invertible 1x1 convolution
class Invertible1x1Conv(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
        objective = objective + dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output, objective

    def reverse_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs().log() * x.size(-2) * x.size(-1)
        objective = objective - dlogdet
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output, objective

    def gradient_(self, out, grad_output):
        with torch.no_grad():
            weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
            x = F.conv2d(out, weight_inv, self.bias, self.stride, self.padding, self.dilation, self.groups)

        with torch.set_grad_enabled(True):
            x.requires_grad = True

            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # perform full backward pass on graph
            dd = torch.autograd.grad(y, (x,) + tuple(self.parameters()), grad_output)

            grad_input = dd[0]
            NN_grad = dd[1:]

            # cleanup sub-graph
            y.detach_()
            del y

        return x, grad_input, reversed(NN_grad)

    def gradient_forward_(self, out, grad_output):
        with torch.no_grad():
            y = F.conv2d(out, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        with torch.set_grad_enabled(True):
            y.requires_grad = True
            weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
            x = F.conv2d(y, weight_inv, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # perform full backward pass on graph
            dd = torch.autograd.grad(x, (y,) + tuple(self.parameters()), grad_output)

            grad_input = dd[0]
            NN_grad = dd[1:]

            # cleanup sub-graph
            x.detach_()
            del x

        return y, grad_input, NN_grad


# ------------------------------------------------------------------------------
# Layers involving squeeze operations defined in RealNVP / Glow.
# ------------------------------------------------------------------------------

# Trades space for depth and vice versa
class Squeeze(Layer):
    def __init__(self, input_shape, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor
        self.input_shape = input_shape

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)
        return x

    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward_(self, x, objective):
        if len(x.size()) != 4:
            raise NotImplementedError

        return self.squeeze_bchw(x), objective

    def reverse_(self, x, objective):
        if len(x.size()) != 4:
            raise NotImplementedError

        return self.unsqueeze_bchw(x), objective

    def gradient_(self, out, grad_output):
        with torch.no_grad():
            x = self.unsqueeze_bchw(out)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x.requires_grad = True

            y = self.squeeze_bchw(x)

            # perform full backward pass on graph
            grad_input = torch.autograd.grad(y, x, grad_output)[0]

            # cleanup sub-graph
            del y

        return x, grad_input, [None]  # None because there are no parameters in this class

    def gradient_forward_(self, out, grad_output):
        with torch.no_grad():
            y = self.squeeze_bchw(out)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y.requires_grad = True

            x = self.unsqueeze_bchw(y)

            # perform full backward pass on graph
            grad_input = torch.autograd.grad(x, y, grad_output)[0]

            # cleanup sub-graph
            del x

        return y, grad_input, [None]  # None because there are no parameters in this class


# ------------------------------------------------------------------------------
# Layers involving prior
# ------------------------------------------------------------------------------

# Gaussian Prior that's compatible with the Layer framework
class GaussianPrior(Layer):
    def __init__(self, input_shape, args):
        super(GaussianPrior, self).__init__()
        self.input_shape = input_shape
        if args.learntop:
            self.conv = Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2)
        else:
            self.conv = None

    def forward_(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)

        if self.conv:
            mean_and_logsd = self.conv(mean_and_logsd)

        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)

        pz = gaussian_diag(mean, logsd)
        objective += pz.logp(x)

        # this way, you can encode and decode back the same image.
        return x, objective

    def reverse_(self, x, objective):
        z = x
        # this way, you can encode and decode back the same image.
        return z, objective

    def gradient_(self, out, grad_output):
        return out, grad_output, [None]

    def gradient_forward_(self, out, grad_output):
        return out, grad_output, [None]


# ------------------------------------------------------------------------------
# Coupling Layers
# ------------------------------------------------------------------------------

# Additive Coupling Layer
class AdditiveCoupling(Layer):
    def __init__(self, num_features, args):
        super(AdditiveCoupling, self).__init__()
        assert num_features % 2 == 0

        self.NN = NN(num_features // 2,
                     hidden_channels=args.hidden_channels,
                     actnorm=args.use_actnorm_in_blocks,
                     an_stable_eps=args.an_stable_eps,
                     relu_inplace=args.relu_inplace,
                     zero_init=args.zero_init)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 = z2 + self.NN(z1)
        out = torch.cat([z1, z2], dim=1)
        return out, objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        z2 = z2 - self.NN(z1)
        out = torch.cat([z1, z2], dim=1)
        return out, objective

    def gradient_(self, out, grad_output):
        y1, y2 = torch.chunk(out, 2, dim=1)

        assert(grad_output.shape[1] % 2 == 0)

        with torch.no_grad():
            # recompute x
            x1 = y1 + 0.0
            x2 = y2 - self.NN(y1)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1.requires_grad = True
            x2.requires_grad = True

            y1 = x1 + 0.0
            y2 = x2 + self.NN(y1)
            y = torch.cat([y1, y2], dim=1)

            # perform full backward pass on graph
            dd = torch.autograd.grad(y, (x1, x2) + tuple(self.NN.parameters()), grad_output)

            NN_grad = dd[2:]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1.detach_()
            y2.detach_()
            del y1, y2

        # restore input
        xout = torch.cat([x1, x2], dim=1).contiguous()

        return xout, grad_input, reversed(NN_grad)

    def gradient_forward_(self, out, grad_output):
        x1, x2 = torch.chunk(out, 2, dim=1)

        assert(grad_output.shape[1] % 2 == 0)

        with torch.no_grad():
            # recompute y
            y1 = x1 + 0.0
            y2 = x2 + self.NN(y1)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y1.requires_grad = True
            y2.requires_grad = True

            x1 = y1 + 0.0
            x2 = y2 - self.NN(x1)
            x = torch.cat([x1, x2], dim=1)

            # perform full backward pass on graph
            dd = torch.autograd.grad(x, (y1, y2) + tuple(self.NN.parameters()), grad_output)

            NN_grad = dd[2:]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            x1.detach_()
            x2.detach_()
            del x1, x2

        # restore input
        yout = torch.cat([y1, y2], dim=1).contiguous()

        return yout, grad_input, NN_grad


class AffineCoupling(Layer):
    def __init__(self, num_features, args):
        super(AffineCoupling, self).__init__()
        self.affine_scale_fn = args.affine_scale_fn
        self.affine_eps = args.affine_eps

        self.NN = NN(num_features // 2,
                     hidden_channels=args.hidden_channels,
                     channels_out=num_features,
                     actnorm=args.use_actnorm_in_blocks,
                     an_stable_eps=args.an_stable_eps,
                     relu_inplace=args.relu_inplace,
                     zero_init=args.zero_init)

    def forward_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]

        if self.affine_scale_fn == 'sigmoid':
            scale = torch.sigmoid(h[:, 1::2] + 2.)
        elif self.affine_scale_fn == 'exp':
            scale = torch.exp(h[:, 1::2])

        z2 = z2 + shift
        z2 = z2 * (scale + self.affine_eps)
        out = torch.cat([z1, z2], dim=1)

        objective = objective + flatten_sum(torch.log(scale))
        return out, objective

    def reverse_(self, x, objective):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]

        if self.affine_scale_fn == 'sigmoid':
            scale = torch.sigmoid(h[:, 1::2] + 2.)
        elif self.affine_scale_fn == 'exp':
            scale = torch.exp(h[:, 1::2])

        z2 = z2 / (scale + self.affine_eps)
        z2 = z2 - shift
        out = torch.cat([z1, z2], dim=1)
        objective = objective - flatten_sum(torch.log(scale))
        return out, objective

    def gradient_(self, out, grad_output):
        y1, y2 = torch.chunk(out, 2, dim=1)

        # partition output gradient also on channels
        assert (grad_output.shape[1] % 2 == 0)

        with torch.set_grad_enabled(False):
            # recompute x
            z1_stop = y1
            h = self.NN(z1_stop)
            shift = h[:, 0::2]
            scale = torch.sigmoid(h[:, 1::2] + 2.)

            x1 = y1 + 0
            x2 = (y2 / (scale + self.affine_eps)) - shift

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1.requires_grad = True
            x2.requires_grad = True

            h = self.NN(x1)
            shift = h[:, 0::2]
            scale = torch.sigmoid(h[:, 1::2] + 2.)

            y1 = x1 + 0
            y2 = x2 + shift
            y2 = y2 * (scale + self.affine_eps)
            y = torch.cat([y1, y2], dim=1)

            # perform full backward pass on graph
            dd = torch.autograd.grad(y, (x1, x2) + tuple(self.NN.parameters()), grad_output)

            NN_grad = dd[2:]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1.detach_()
            y2.detach_()
            del y1, y2

        # restore input
        xout = torch.cat([x1, x2], dim=1).contiguous()
        return xout, grad_input, reversed(NN_grad)

    def gradient_forward_(self, out, grad_output):
        x1, x2 = torch.chunk(out, 2, dim=1)

        assert(grad_output.shape[1] % 2 == 0)

        with torch.no_grad():
            # recompute y
            z1_stop = x1
            h = self.NN(z1_stop)
            shift = h[:, 0::2]
            scale = torch.sigmoid(h[:, 1::2] + 2.)

            y1 = x1 + 0
            y2 = x2 + shift
            y2 = y2 * (scale + self.affine_eps)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y1.requires_grad = True
            y2.requires_grad = True

            h = self.NN(y1)
            shift = h[:, 0::2]
            scale = torch.sigmoid(h[:, 1::2] + 2.)

            x1 = y1 + 0
            x2 = y2 / (scale + self.affine_eps)
            x2 = x2 - shift
            x = torch.cat([x1, x2], dim=1)

            # perform full backward pass on graph
            dd = torch.autograd.grad(x, (y1, y2) + tuple(self.NN.parameters()), grad_output)

            NN_grad = dd[2:]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            x1.detach_()
            x2.detach_()
            del x1, x2

        # restore input
        yout = torch.cat([y1, y2], dim=1).contiguous()

        return yout, grad_input, NN_grad


# ------------------------------------------------------------------------------
# Normalizing Layers
# ------------------------------------------------------------------------------

# ActNorm Layer with data-dependant init
class ActNorm(Layer):
    def __init__(self, num_features, logscale_factor=1., scale=1., stable_eps=0):
        super(Layer, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.stable_eps = stable_eps

    def forward_(self, input, objective):
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)  # Compresses the two spatial dimensions into 1 (512, 1024, 1)

        if not self.initialized:
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            b = -torch.sum(input, dim=(0, -1)) / sum_size  # Computes the (negative) mean of the input features
            vars = unsqueeze(torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1)) / (sum_size-1))
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6)) / self.logscale_factor

            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b
        output = (input + b) * (torch.exp(logs) +  self.stable_eps)
        dlogdet = torch.sum(torch.log(torch.exp(self.logs * self.logscale_factor) +  self.stable_eps)) * input.size(-1) # c x h
        return output.view(input_shape), objective + dlogdet

    def reverse_(self, input, objective):
        assert self.initialized

        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        output = input / (torch.exp(logs) + self.stable_eps) - b
        dlogdet = torch.sum(torch.log(torch.exp(logs) +  self.stable_eps)) * input.size(-1) # c x h

        return output.view(input_shape), objective - dlogdet

    def gradient_(self, out, grad_output):
        with torch.set_grad_enabled(False):
            orig_shape = out.size()
            x = out.view(orig_shape[0], orig_shape[1], -1)
            x = x / (torch.exp(self.logs * self.logscale_factor) + self.stable_eps) - self.b

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x.requires_grad = True
            y = (x + self.b) * (torch.exp(self.logs * self.logscale_factor) + self.stable_eps)

            y = y.view(orig_shape)

            # perform full backward pass on graph
            dd = torch.autograd.grad(y, (x, self.logs, self.b), grad_output)

            x = x.view(orig_shape)
            grad_input = dd[0].view(orig_shape)
            grad_logs = dd[1]
            grad_b = dd[2]

            del y

        with torch.no_grad():
            xout = x + 0

        return xout, grad_input, reversed([grad_b, grad_logs])

    def gradient_forward_(self, out, grad_output):
        with torch.set_grad_enabled(False):
            orig_shape = out.size()
            y = out.view(orig_shape[0], orig_shape[1], -1)
            y = (y + self.b) * (torch.exp(self.logs * self.logscale_factor) + self.stable_eps)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y.requires_grad = True
            x = y / (torch.exp(self.logs * self.logscale_factor) + self.stable_eps) - self.b

            x = x.view(orig_shape)

            # perform full backward pass on graph
            dd = torch.autograd.grad(x, (y, self.logs, self.b), grad_output)

            y = y.view(orig_shape)
            grad_input = dd[0].view(orig_shape)
            grad_logs = dd[1]
            grad_b = dd[2]

            del x

        with torch.no_grad():
            yout = y + 0

        return yout, grad_input, [grad_b, grad_logs]


# ------------------------------------------------------------------------------
# Stacked Layers
# ------------------------------------------------------------------------------
class RevNetStep(LayerList):
    def __init__(self, num_channels, use_actnorm, args):
        super(RevNetStep, self).__init__()
        self.args = args
        layers = []
        if use_actnorm:
            layers += [ActNorm(num_channels, stable_eps=args.an_stable_eps)]

        if args.permutation == 'reverse':
            layers += [Reverse(num_channels)]
        elif args.permutation == 'shuffle':
            layers += [Shuffle(num_channels)]
        elif args.permutation == 'conv':
            layers += [Invertible1x1Conv(num_channels)]
        else:
            raise ValueError

        if args.coupling == 'additive':
            layers += [AdditiveCoupling(num_channels, args)]
        elif args.coupling == 'affine':
            layers += [AffineCoupling(num_channels, args)]
        else:
            raise ValueError
        self.layers = nn.ModuleList(layers)


# Full model
class Glow_(LayerList, nn.Module):
    def __init__(self, input_shape, args):
        super(Glow_, self).__init__()
        layers = []
        output_shapes = []

        self.ipad = args.inj_pad
        self.inj_pad = injective_pad(args.inj_pad)

        # Adjust input_shape to take into account padding
        input_shape = (input_shape[0], input_shape[1] + self.ipad, input_shape[2], input_shape[3])
        _, C, H, W = input_shape

        for i in range(args.n_levels):
            # Squeeze Layer
            layers += [Squeeze(input_shape)]
            C, H, W = C * 4, H // 2, W // 2

            # RevNet Block
            if H == 1 and W == 1:  # If feature maps have size 1x1
                layers += [RevNetStep(C, use_actnorm=False, args=args) for _ in range(args.depth)]
            else:
                layers += [RevNetStep(C, use_actnorm=args.use_actnorm, args=args) for _ in range(args.depth)]

        if args.use_prior:
            layers += [GaussianPrior((args.batch_size, C, H, W), args)]

        self.layers = nn.ModuleList(layers)
        self.output_shapes = output_shapes
        self.args = args
        self.flatten()

    def forward(self, *inputs):
        x, objective = inputs

        if self.ipad != 0:
            x = self.inj_pad.forward(x)

        return self.forward_(x, objective)

    def sample(self, x, no_grad=True):
        if no_grad:
            with torch.no_grad():
                samples = self.reverse_(x, 0.)[0]
                if self.ipad != 0:
                    samples = self.inj_pad.inverse(samples)
                return samples
        else:
            samples = self.reverse_(x, 0.)[0]
            if self.ipad != 0:
                samples = self.inj_pad.inverse(samples)
            return samples

    def gradient(self, out, grad_output, return_grad_input=False):
        grad_input, gradients = self.gradient_(out, grad_output)
        if return_grad_input:
            return grad_input, gradients
        else:
            return gradients

    def gradient_forward(self, out, grad_output, out_ForwPass, noise_dir, noise_eps, return_grad_input=False):
        grad_input, gradients, out = self.gradient_forward_(out, grad_output, out_ForwPass, noise_dir, noise_eps)
        if return_grad_input:
            return grad_input, gradients
        else:
            return gradients, out

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
