"""Wittman style plot for the report"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_normal

# Make TeX labels work on plots
#plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
#plt.rc('text', usetex=True)

# Grab some data
data_train = pd.read_csv('./final_run_data/data_training.csv')
data_valid_0 = pd.read_csv('./final_run_data/different_sn/data_validation_sn=1.0.csv')
data_valid_4 = pd.read_csv('./final_run_data/different_sn/data_validation_sn=4.0.csv')

results_valid_0 = pd.read_csv('./final_run_data/different_sn/validation_results_sn=1.0.csv')
results_valid_4 = pd.read_csv('./final_run_data/different_sn/validation_results_sn=4.0.csv')

with open('./final_run_data/different_sn/validation_mixtures_sn=1.0.pkl', 'rb') as f:
    mixtures_0 = pickle.load(f)

with open('./final_run_data/different_sn/validation_mixtures_sn=4.0.pkl', 'rb') as f:
    mixtures_4 = pickle.load(f)

# Make a Wittman+2014-style plot
fig = plt.figure(figsize=(3, 3))
ax_wittman = fig.add_subplot(1, 1, 1)

# Just in case you passed a data frame or some bullshit like that
spectroscopic_z = np.asarray(data_valid_0['z_spec']).flatten()
validation_mixtures = mixtures_0
validation_results = results_valid_0

# Calculate cdfs of all mixtures at all ma points!
cdfs = scipy_normal.cdf(np.tile(spectroscopic_z, (validation_mixtures['means'].shape[1], 1)).T,
                        loc=validation_mixtures['means'],
                        scale=validation_mixtures['std_deviations'])

# Multiply by the weights and sum
cdfs = np.sum(validation_mixtures['weights'] * cdfs, axis=1)

# Normalise the cdfs with the initial and end values
cdfs = cdfs * validation_results['cdf_multiplier'] + validation_results['cdf_constant']

cdfs_sorted = np.sort(cdfs)

cdfs_summed = np.linspace(0., 1., num=cdfs_sorted.size)

residual = np.abs(cdfs_summed - cdfs_sorted)
max_residual_0 = np.max(residual)

ax_wittman.plot([0, 1], [0, 1], 'k-')

ax_wittman.plot(cdfs_sorted, cdfs_summed, 'r-', label=r'$SNR \cdot 1.00$', alpha=1.0)

#####################################

# Just in case you passed a data frame or some bullshit like that
spectroscopic_z = np.asarray(data_valid_4['z_spec']).flatten()
validation_mixtures = mixtures_4
validation_results = results_valid_4

# Calculate cdfs of all mixtures at all ma points!
cdfs = scipy_normal.cdf(np.tile(spectroscopic_z, (validation_mixtures['means'].shape[1], 1)).T,
                        loc=validation_mixtures['means'],
                        scale=validation_mixtures['std_deviations'])

# Multiply by the weights and sum
cdfs = np.sum(validation_mixtures['weights'] * cdfs, axis=1)

# Normalise the cdfs with the initial and end values
cdfs = cdfs * validation_results['cdf_multiplier'] + validation_results['cdf_constant']

cdfs_sorted = np.sort(cdfs)

cdfs_summed = np.linspace(0., 1., num=cdfs_sorted.size)

residual = np.abs(cdfs_summed - cdfs_sorted)
max_residual_4 = np.max(residual)

ax_wittman.plot(cdfs_sorted, cdfs_summed, 'b--', label=r'$SNR \cdot 0.25$', alpha=1.0)

#####################################

ax_wittman.text(0.35, 0.02,
        'KS test at:           \n' + r'$SNR \cdot 1.00$' + ' = {:.5f}'.format(max_residual_0)
        + '\n' + r'$SNR \cdot 0.25$' + ' = {:.5f}'.format(max_residual_4),
        ha='left', va='bottom', transform=ax_wittman.transAxes, fontsize=8,)
        # bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0), ))

# Labels
#ax_wittman.set_xlabel(r'$c_i$')
#ax_wittman.set_ylabel(r'$F(c_i)$')
ax_wittman.set_xlabel('Confidence interval')
ax_wittman.set_ylabel('Cumulative sum of conf. intervals')
ax_wittman.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)
ax_wittman.set_xlim(0, 1)
ax_wittman.set_ylim(0, 1)

fig.tight_layout()
fig.savefig('./final_plots/pres_wittman.png', dpi=600)

fig.show()
