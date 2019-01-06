"""PDF plots for the report"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_normal

# Make TeX labels work on plots
plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
plt.rc('text', usetex=True)

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

# Make some pwetty PDF plots
fig = plt.figure(figsize=(3, 3))
ax_1 = fig.add_subplot(2, 2, 1)
ax_2 = fig.add_subplot(2, 2, 3, sharex=ax_1)
ax_3 = fig.add_subplot(2, 2, 2)
ax_4 = fig.add_subplot(2, 2, 4, sharex=ax_3)

ax_pdf_0 = ax_1
ax_pdf_4 = ax_2

# Galaxy to plot
galaxy = 7  # 12, 39, 45, 18
resolution = 200
range = [1.7, 4.7]
x_range = np.linspace(range[0], range[1], num=200)

# list of colours
colors = ['r', 'y', 'c', 'b', 'm']

# Get its deets
z_spec = data_valid_0.loc[galaxy, 'z_spec']
z_ml_0 = results_valid_0['map']
z_ml_4 = results_valid_4['map']
z_upper_0 = results_valid_0['upper']
z_upper_4 = results_valid_4['upper']
z_lower_0 = results_valid_0['lower']
z_lower_4 = results_valid_4['lower']

#######################
# SN0 plot galaxy 1

# Keep a record of the running total
total = np.zeros(resolution)

# Grab our deets
means = mixtures_0['means'][galaxy]
std_deviations = mixtures_0['std_deviations'][galaxy]
weights = mixtures_0['weights'][galaxy]

# Plot all the mixtures iteratively
i = 1
for a_mean, a_std, a_weight, a_color in zip(means, std_deviations, weights, colors):
    this_pdf = a_weight * scipy_normal.pdf(x_range, loc=a_mean, scale=a_std)
    total = total + this_pdf
    ax_pdf_0.fill_between(x_range, this_pdf, color=a_color, alpha=0.3,
                  label='Mixture {}'.format(i))
    i += 1

y_size = total.max() * 1.1

ax_pdf_0.plot(x_range, total, '-', color='k', lw=1, alpha=1.0, label='Summed mixtures')
ax_pdf_0.plot([z_spec, z_spec], [0, y_size], 'r-', alpha=1.0, lw=1, label='True value')

# Labels and stuff
ax_pdf_0.set_xlim(0, 7)
ax_pdf_0.set_ylim(0, total.max() * 1.1)

ax_pdf_0.set_ylabel(r'$P(z)$')

#######################
# SN4 plot galaxy 1

# Keep a record of the running total
total = np.zeros(resolution)

# Grab our deets
means = mixtures_4['means'][galaxy]
std_deviations = mixtures_4['std_deviations'][galaxy]
weights = mixtures_4['weights'][galaxy]

# Plot all the mixtures iteratively
for a_mean, a_std, a_weight, a_color in zip(means, std_deviations, weights, colors):
    this_pdf = a_weight * scipy_normal.pdf(x_range, loc=a_mean, scale=a_std)
    total = total + this_pdf
    ax_pdf_4.fill_between(x_range, this_pdf, color=a_color, alpha=0.3,
                          label='Mixture {}'.format(i))

y_size = total.max() * 1.1

ax_pdf_4.plot(x_range, total, '-', color='k', lw=1, alpha=1.0, label='Summed total')
ax_pdf_4.plot([z_spec, z_spec], [0, y_size], 'r-', alpha=1.0, lw=1)

# Labels and stuff
ax_pdf_4.set_xlim(range[0], range[1])
ax_pdf_4.set_ylim(0, y_size)

###############################################################################

ax_pdf_0 = ax_3
ax_pdf_4 = ax_4

# Galaxy to plot
galaxy = 12  # 7, 12, 26, 35
resolution = 200
range = [0, 3.1]
x_range = np.linspace(range[0], range[1], num=200)

# list of colours
colors = ['r', 'y', 'c', 'b', 'm']

# Get its deets
z_spec = data_valid_0.loc[galaxy, 'z_spec']
z_ml_0 = results_valid_0['map']
z_ml_4 = results_valid_4['map']
z_upper_0 = results_valid_0['upper']
z_upper_4 = results_valid_4['upper']
z_lower_0 = results_valid_0['lower']
z_lower_4 = results_valid_4['lower']

#######################
# SN0 plot galaxy 2

# Keep a record of the running total
total = np.zeros(resolution)

# Grab our deets
means = mixtures_0['means'][galaxy]
std_deviations = mixtures_0['std_deviations'][galaxy]
weights = mixtures_0['weights'][galaxy]

# Plot all the mixtures iteratively
i = 1
for a_mean, a_std, a_weight, a_color in zip(means, std_deviations, weights, colors):
    this_pdf = a_weight * scipy_normal.pdf(x_range, loc=a_mean, scale=a_std)
    total = total + this_pdf
    ax_pdf_0.fill_between(x_range, this_pdf, color=a_color, alpha=0.3,
                  label='Mixture {}'.format(i))
    i += 1

y_size = total.max() * 1.1

ax_pdf_0.plot(x_range, total, '-', color='k', lw=1, alpha=1.0, label='Summed mixtures')
ax_pdf_0.plot([z_spec, z_spec], [0, y_size], 'r-', alpha=1.0, lw=1, label='True value')

# Labels and stuff
ax_pdf_0.set_xlim(range[0], range[1])
ax_pdf_0.set_ylim(0, y_size)

#######################
# SN4 plot galaxy 2

# Keep a record of the running total
total = np.zeros(resolution)

# Grab our deets
means = mixtures_4['means'][galaxy]
std_deviations = mixtures_4['std_deviations'][galaxy]
weights = mixtures_4['weights'][galaxy]

# Plot all the mixtures iteratively
for a_mean, a_std, a_weight, a_color in zip(means, std_deviations, weights, colors):
    this_pdf = a_weight * scipy_normal.pdf(x_range, loc=a_mean, scale=a_std)
    total = total + this_pdf
    ax_pdf_4.fill_between(x_range, this_pdf, color=a_color, alpha=0.3,
                          label='Mixture {}'.format(i))

y_size = total.max() * 1.1

ax_pdf_4.plot(x_range, total, '-', color='k', lw=1, alpha=1.0, label='Summed total')
ax_pdf_4.plot([z_spec, z_spec], [0, y_size], 'r-', alpha=1.0, lw=1)

# Labels
ax_pdf_4.set_xlim(range[0], range[1])
ax_pdf_4.set_ylim(0, y_size)

#######################

# Final deets and saving

ax_1.set_ylabel(r'$P(z)$')
ax_2.set_ylabel(r'$P(z)$')
ax_2.set_xlabel(r'$z$')
ax_4.set_xlabel(r'$z$')

ax_1.set_title('Galaxy 7', fontsize=8)
ax_3.set_title('Galaxy 12', fontsize=8)

# Label plots
ax_3.text(-0.03, 0.8, r'$SNR \cdot 1.0$',
          ha='center', va='center', transform=ax_3.transAxes, fontsize=8,
          bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0), ))

ax_4.text(-0.03, 0.8, r'$SNR \cdot 0.25$',
          ha='center', va='center', transform=ax_4.transAxes, fontsize=8,
          bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0), ))

# Add a legend to plot 3
#ax_3.legend(edgecolor='k', facecolor='w', fancybox=False, fontsize=8,
#            bbox_to_anchor=(0.0, 1.0), ncols=3)

# Fix possible spacing issues between left and right subplots
fig.subplots_adjust(hspace=0, wspace=0.05)

# Sort out the ticks
ax_2.set_xticks([2, 3, 4])
ax_4.set_xticks([0, 1, 2, 3])
ax_4.set_yticks([0, 1, 2])

ax_3.yaxis.tick_right()
ax_4.yaxis.tick_right()
plt.setp([a.get_xticklabels() for a in fig.axes[::2]], visible=False)

# fig.tight_layout()

fig.savefig('./final_plots/pdfs.png', dpi=600, bbox_inches='tight')

fig.show()

