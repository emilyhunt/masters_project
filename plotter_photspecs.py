"""Makes all of the zphot vs zspec plots for ma reporty mc reportface."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts import z_plot


# Grab some data
data_train = pd.read_csv('./final_run_data/data_training.csv')
data_valid = pd.read_csv('./final_run_data/data_validation.csv')
data_test = pd.read_csv('./final_run_data/data_no_spec_z.csv')

results_valid = pd.read_csv('./final_run_data/validation_results.csv')
results_test = pd.read_csv('./final_run_data/validation_results_no_spec_z.csv')


# VALIDATION DATA PLOTS
# For ML
sigma_3, sigma_5 = z_plot.error_evaluator(data_valid['z_spec'], results_valid['map'],
                                          results_valid['lower'], results_valid['upper'])

z_plot.phot_vs_spec(data_valid['z_spec'], results_valid['map'], nmad_bins=10, limits=[0, 7], show_5_sigma=sigma_5/100,
                    save_name='./final_plots/ML_valid_phot_spec.png', show_fig=True, set_nmad_old_value=True)

# For EAZY
sigma_3, sigma_5 = z_plot.error_evaluator(data_valid['z_spec'], data_valid['z_phot_lit'],
                                          data_valid['z_phot_lit_l68'], data_valid['z_phot_lit_u68'])

z_plot.phot_vs_spec(data_valid['z_spec'], data_valid['z_phot_lit'], nmad_bins=10, limits=[0, 7], show_5_sigma=sigma_5/100,
                    save_name='./final_plots/EAZY_valid_phot_spec.png', show_fig=True, point_color='b',
                    x_label=r'$z_{spec}$', y_label=r'$z_{EAZY}$')

# For test data
sigma_3, sigma_5 = z_plot.error_evaluator(data_test['z_phot_lit'], results_test['map'],
                                          results_test['lower'], results_test['upper'])

z_plot.phot_vs_spec(data_test['z_phot_lit'], results_test['map'], nmad_bins=10, limits=[0, 7], show_5_sigma=sigma_5/100,
                    save_name='./final_plots/ML_EAZY_test_phot_spec.png', show_fig=True, point_color='b', point_alpha=0.02,
                    plot_extra_axes=False, figsize=(3, 3),
                    x_label=r'$z_{EAZY}$', y_label=r'$z_{ML}$')
