"""
Helper script to analyze results on 2D checkerboard data.
"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors


def masked_data(grid_data):
    dim = int(np.sqrt(grid_data.shape[0]))
    grid_data = grid_data.reshape(dim, dim)
    grid_data = np.ma.masked_where(np.isnan(grid_data), grid_data)
    grid_data = np.ma.masked_where(~np.isfinite(grid_data), grid_data)
    return grid_data

domain = 8
resolution = 0.05
x = np.arange(-domain, domain, resolution)
y = np.arange(-domain, domain, resolution)
xx, yy = np.meshgrid(x, y, sparse=False)

path_m1 = 'experiments/iresnet_toyL2_08_100blocks/'
name_m1 = 'iresnet_coeff08_100blocks'
recon_error_m1 = np.load(path_m1 + 'recon-error-area8_05.npy')

path_m2 = 'experiments/realnvp_toy_100blocks/'
name_m2 = 'affine_100blocks'
recon_error_m2 = np.load(path_m2 + 'recon-error-area8_05.npy')

path_m3 = 'experiments/realnvp_toy_sigmoid05_100blocks/'
name_m3 = 'affine_sigmoid05_100blocks'
recon_error_m3 = np.load(path_m3 + 'recon-error-area8_05.npy')

# mask data
recon_error_m1_mask = masked_data(recon_error_m1)
recon_error_m2_mask = masked_data(recon_error_m2)
recon_error_m3_mask = masked_data(recon_error_m3)

# find min and max values
min_plot = np.min([recon_error_m1_mask.min(),
                   recon_error_m2_mask.min(),
                   recon_error_m3_mask.min()])
max_plot = np.max([recon_error_m1_mask.max(),
                   recon_error_m2_mask.max(),
                   recon_error_m3_mask.max()])
if min_plot <= 1e-8:
    min_plot = 1e-8
recon_error_m1_mask[recon_error_m1_mask <= min_plot] = 1e-8
recon_error_m2_mask[recon_error_m2_mask <= min_plot] = 1e-8
recon_error_m3_mask[recon_error_m3_mask <= min_plot] = 1e-8
if max_plot >= 1:
    max_plot = 1
recon_error_m1_mask[recon_error_m1_mask >= max_plot] = 1
recon_error_m2_mask[recon_error_m2_mask >= max_plot] = 1
recon_error_m3_mask[recon_error_m3_mask >= max_plot] = 1

fig, axs = plt.subplots(1, 3)
im1 = axs[0].pcolormesh(xx, yy, recon_error_m1_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[0].set_title(name_m1)
im2 = axs[1].pcolormesh(xx, yy, recon_error_m2_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[1].set_title(name_m2)
im3 = axs[2].pcolormesh(xx, yy, recon_error_m3_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[2].set_title(name_m3)
#fig.colorbar(im1, ax=axs[0])
#fig.colorbar(im2, ax=axs[1])
#fig.colorbar(im3, ax=axs[2])

fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, recon_error_m1_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
fig.colorbar(im, ax=ax1)

fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, recon_error_m2_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
fig.colorbar(im, ax=ax1)

fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, recon_error_m3_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
fig.colorbar(im, ax=ax1)


# plot checkerboard
# checkerboard data
def checkerboard(batch_size):
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

p_samples = checkerboard(batch_size=200000)

LOW = -8
HIGH = 8

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
npts=100
ax1.hist2d(p_samples[:, 0], p_samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, cmap='inferno')



### plot likelihood
likel_m1 = np.load(path_m1 + 'likelihood-area8_05.npy')
likel_m2 = np.load(path_m2 + 'likelihood-area8_05.npy')
likel_m3 = np.load(path_m3 + 'likelihood-area8_05.npy')

likel_m1_mask = masked_data(likel_m1)
likel_m2_mask = masked_data(likel_m2)
likel_m3_mask = masked_data(likel_m3)

if min_plot <= 1e-8:
    min_plot = 1e-8
likel_m1_mask[likel_m1_mask <= min_plot] = 1e-8
likel_m1_mask[likel_m1_mask <= min_plot] = 1e-8
likel_m3_mask[likel_m3_mask <= min_plot] = 1e-8
if max_plot >= 1e5:
    max_plot = 1e5
likel_m1_mask[likel_m1_mask >= max_plot] = 1e5
likel_m2_mask[likel_m2_mask >= max_plot] = 1e5
likel_m3_mask[likel_m3_mask >= max_plot] = 1e5

fig, axs = plt.subplots(1, 3)
im1 = axs[0].pcolormesh(xx, yy, likel_m1_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[0].set_title(name_m1)
im2 = axs[1].pcolormesh(xx, yy, likel_m2_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[1].set_title(name_m2)
im3 = axs[2].pcolormesh(xx, yy, likel_m3_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
axs[2].set_title(name_m3)


fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, likel_m1_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')

fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, likel_m2_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')

fig1, ax1 = plt.subplots()
im = ax1.pcolormesh(xx, yy, likel_m3_mask,
               norm=colors.LogNorm(vmin=min_plot, vmax=max_plot),
               cmap='inferno')
