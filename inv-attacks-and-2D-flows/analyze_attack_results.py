"""
Helper script to analyze results from invertibility attack
"""

import numpy as np
import matplotlib.pylab as plt

path_m1 = 'experiments/affineCelebA64/'
name_m1 = 'affineCelebA64'
recon_error_m1 = np.load(path_m1 + 'recon_errors.npy')

path_m2 = 'experiments/affineCelebA64_05sigmoid/'
name_m2 = 'affineCelebA64_05sigmoid'
recon_error_m2 = np.load(path_m2 + 'recon_errors.npy')

path_m3 = 'experiments/resflowCelebA64/'
name_m3 = 'resflowCelebA64'
recon_error_m3 = np.load(path_m3 + 'recon_errors.npy')

# normalize error with image size
recon_error_m1 = recon_error_m1 / (64*64*3)
recon_error_m2 = recon_error_m2 / (64*64*3)
recon_error_m3 = recon_error_m3 / (64*64*3)

# find NaN values
recon_error_m1_nan = np.isnan(recon_error_m1)
i_m1 = 56
max_plot = np.nanmax([np.nanmax(recon_error_m1), np.nanmax(recon_error_m2)])

recon_error_m2_nan = np.isnan(recon_error_m2)
i_m2 = 143

min_plot = np.nanmin([recon_error_m1, recon_error_m2])

fig, ax = plt.subplots()
im = ax.semilogy(recon_error_m1, 'k', linewidth=3)
im = ax.semilogy(recon_error_m2, 'r', linewidth=3)
im = ax.semilogy(recon_error_m3, 'b', linewidth=3)
im = ax.plot(i_m2, max_plot, 'rx', linewidth=3)
im = ax.plot(i_m1, max_plot, 'kx', linewidth=3)
ax.tick_params(labelsize=16)
plt.ylim([min_plot,max_plot+1e-3])
plt.legend(['Affine', 'mod. Affine', 'Resflow'], prop={'size': 14})
