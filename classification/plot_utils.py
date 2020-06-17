"""Plotting utilities.
"""
import os
import csv
import ipdb
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


import seaborn as sns
sns.set(style='white')


def load_epoch_log(exp_dir):
    epoch_dict = defaultdict(list)
    with open(os.path.join(exp_dir, 'epoch_log.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                epoch_dict[key].append(float(row[key]))
    return epoch_dict


def load_log(exp_dir, filename='every_N_log.csv'):
    result_dict = defaultdict(list)
    with open(os.path.join(exp_dir, filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                if not row[key] == '':
                    result_dict[key].append(float(row[key]))
    return result_dict


def load_iteration_log(exp_dir):
    result_dict = defaultdict(list)
    with open(os.path.join(exp_dir, 'iteration_log.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result_dict['global_iteration'].append(int(row['global_iteration']))
            result_dict['loss'].append(float(row['loss']))

    return result_dict


def plot_stability_stats(exp_dir, filename='svd_log.csv'):
    result_dict = load_log(exp_dir, filename)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14,3))

    ax[0].plot(result_dict['global_iteration'], result_dict['mean_recons_error'], linewidth=2)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Iteration', fontsize=18)
    ax[0].set_ylabel('Recons Error', fontsize=18)

    if 'condition_num' in result_dict:
        ax[1].plot(result_dict['global_iteration'], result_dict['condition_num'], linewidth=2)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Iteration', fontsize=18)
        ax[1].set_ylabel('Condition Num', fontsize=18)

        ax[2].plot(result_dict['global_iteration'], result_dict['max_sv'], linewidth=2)
        ax[2].set_yscale('log')
        ax[2].set_xlabel('Iteration', fontsize=18)
        ax[2].set_ylabel('Max singular Value', fontsize=18)

        ax[3].plot(result_dict['global_iteration'], result_dict['min_sv'], linewidth=2)
        ax[3].set_yscale('log')
        ax[3].set_xlabel('Iteration', fontsize=18)
        ax[3].set_ylabel('Min singular Value', fontsize=18)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'stability_stats.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if 'inverse_condition_num' in result_dict:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))

        ax[0].plot(result_dict['global_iteration'], result_dict['inverse_condition_num'], linewidth=2)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Iteration', fontsize=18)
        ax[0].set_ylabel('Inv Condition Num', fontsize=18)

        ax[1].plot(result_dict['global_iteration'], result_dict['inverse_max_sv'], linewidth=2)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Iteration', fontsize=18)
        ax[1].set_ylabel('Inv max singular Value', fontsize=18)

        ax[2].plot(result_dict['global_iteration'], result_dict['inverse_min_sv'], linewidth=2)
        ax[2].set_yscale('log')
        ax[2].set_xlabel('Iteration', fontsize=18)
        ax[2].set_ylabel('Inv min singular Value', fontsize=18)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'stability_stats_inverse.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def plot_item(result_dict,
              xkey,
              ykey,
              xlabel='',
              ylabel='',
              xlabel_fontsize=22,
              ylabel_fontsize=22,
              xtick_fontsize=18,
              ytick_fontsize=18,
              yscale='linear',
              linewidth=2,
              save_to=None):

    fig = plt.figure(figsize=(5,4))

    plt.plot(result_dict[xkey], result_dict[ykey], linewidth=linewidth)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.yscale(yscale)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


def plot_loss(exp_dir):
    figure_dir = os.path.join(exp_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    result_dict = load_iteration_log(exp_dir)

    plot_item(result_dict,
              xkey='global_iteration',
              ykey='loss',
              xlabel='Iteration',
              ylabel='Loss',
              linewidth=3,
              yscale='linear',
              save_to=os.path.join(figure_dir, 'loss-plot.pdf')
             )


def plot_individual_figures(exp_dir, filename='every_N_log.csv'):
    figure_dir = os.path.join(exp_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    result_dict = load_log(exp_dir, filename)

    if 'mean_recons_error' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='mean_recons_error',
                  xlabel='Iteration',
                  ylabel='Reconstruction Error',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'recons-plot.png')
                 )

    if 'condition_num' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='condition_num',
                  xlabel='Iteration',
                  ylabel='Condition Num',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'condition-plot.png')
                 )

    if 'min_sv' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='min_sv',
                  xlabel='Iteration',
                  ylabel='Minimum SV',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'min-sv-plot.png')
                 )

    if 'max_sv' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='max_sv',
                  xlabel='Iteration',
                  ylabel='Maximum SV',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'max-sv-plot.png')
                 )

    if 'inverse_condition_num' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='inverse_condition_num',
                  xlabel='Iteration',
                  ylabel='Inverse Cond. Num',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'inv-condition-plot.png')
                 )

    if 'inverse_max_sv' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='inverse_max_sv',
                  xlabel='Iteration',
                  ylabel='Inverse Max SV',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'inv-max-sv-plot.png')
                 )

    if 'inverse_min_sv' in result_dict:
        plot_item(result_dict,
                  xkey='global_iteration',
                  ykey='inverse_min_sv',
                  xlabel='Iteration',
                  ylabel='Inverse Min SV',
                  linewidth=2,
                  yscale='log',
                  save_to=os.path.join(figure_dir, 'inv-min-sv-plot.png')
                 )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str,
                        help='Path to the experiment directory')
    args = parser.parse_args()

    plot_individual_figures(args.exp_dir)
    plot_loss(args.exp_dir)
