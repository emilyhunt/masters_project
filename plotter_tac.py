"""Makes the plots for the TAC data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make TeX labels work on plots
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})

# Grab the shit
data_sn0 = pd.read_csv('./plots/18-12-24_tac_2/size_study_tseed_sn=0.0.csv')
data_sn0.drop(columns='Unnamed: 0', inplace=True)

# Process the stuff into means at points
points_to_plot = np.unique(data_sn0['dataset_size'])

plot_sn0 = pd.DataFrame({'size': points_to_plot, 'nmad': 0., 'e_nmad': 0., 'sigma_5': 0., 'e_sigma_5': 0., 'n': 0.})

for j, a_point in enumerate(points_to_plot):
    ma_bois = np.where(data_sn0['dataset_size'] == a_point)[0]

    plot_sn0.loc[j, 'nmad'] = np.mean(data_sn0.loc[ma_bois, 'nmad'])
    plot_sn0.loc[j, 'e_nmad'] = np.std(data_sn0.loc[ma_bois, 'nmad'])
    plot_sn0.loc[j, 'sigma_5'] = np.mean(data_sn0.loc[ma_bois, 'sigma_5']) / 100
    plot_sn0.loc[j, 'e_sigma_5'] = np.std(data_sn0.loc[ma_bois, 'sigma_5']) / 100
    plot_sn0.loc[j, 'n'] = ma_bois.size

# Start up matplotlib's engines of death
fig = plt.figure(figsize=(4, 3))
ax_nmad = fig.add_subplot(1, 1, 1)
#ax_nmad.set_yscale('log')

# Plot the nmad
nmad_plot = ax_nmad.errorbar(plot_sn0['size'], plot_sn0['nmad'], yerr=plot_sn0['e_nmad'], fmt='rs-', capsize=2., ms=4, mew=0.5, mec='k',
                 label='Scatter ' + r'($\sigma_{NMAD}$)')
ax_nmad.set_xlabel('Training set size')
# ax_nmad.set_ylabel(r'$\sigma_{NMAD}$')
ax_nmad.set_ylabel('Scatter ' + r'($\sigma_{NMAD}$)')

# Plot the 5 sigma outlier fraction
ax_sigma = ax_nmad.twinx()
sigma_plot = ax_sigma.errorbar(plot_sn0['size'], plot_sn0['sigma_5'], yerr=plot_sn0['e_sigma_5'], fmt='bo-', capsize=2., ms=4, mew=0.5, mec='k',
                  label='Fraction of 5 sigma outliers')
#ax_sigma.set_ylabel(r'$\mathcal{F}_{5\sigma}$')
ax_sigma.set_ylabel('Fraction of 5 sigma outliers')

# Make the legend (we do a fancy thing because of how we have two axes in the same hecking place)
lines, labels = ax_nmad.get_legend_handles_labels()
lines2, labels2 = ax_sigma.get_legend_handles_labels()
ax_sigma.legend(lines + lines2, labels + labels2, edgecolor='k', facecolor='w', fancybox=True, fontsize=8)

fig.tight_layout()
fig.savefig('./final_plots/pres_tac.png', dpi=600)

fig.show()













