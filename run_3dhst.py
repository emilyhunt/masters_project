"""Trains a network on the real 3dhst good south data."""

import numpy as np
import os
import pandas as pd
import time
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scripts import galaxy_pairs
from scripts import twitter
from data import hst_goodss_config
from sklearn.model_selection import train_test_split


# Begin by reading in the data
data = util.read_fits('./data/goodss_3dhst.v4.1.cat.FITS', hst_goodss_config.keys_to_keep,
                      new_column_names=None, get_flux_and_error_keys=False)

archival_redshifts = util.read_save('./data/KMOS415_output/GS415.3dhst.redshift.save')

data[['z_spec', 'z_phot_lit', 'z_phot_lit_l68', 'z_phot_lit_u68']] = \
    archival_redshifts[['gs4_zspec', 'gs4_zphot', 'gs4_zphot_l68', 'gs4_zphot_u68']]

# Get some useful things from the config file
flux_keys = hst_goodss_config.flux_keys_in_order
error_keys = hst_goodss_config.error_keys_in_order
band_central_wavelengths = hst_goodss_config.band_central_wavelengths

# Take a look at the coverage in different photometric bands
data_with_spec_z, data_no_spec_z, reduced_flux_keys, reduced_error_keys = \
    util.check_photometric_coverage_3dhst(data, flux_keys, error_keys, band_central_wavelengths,
                                          coverage_minimum=0.5,
                                          valid_photometry_column='use_phot',
                                          missing_flux_handling='normalised_column_mean',
                                          missing_error_handling='big_value')

data_with_spec_z = util.convert_to_log_sn_errors(data_with_spec_z, reduced_flux_keys, reduced_error_keys)
data_with_spec_z = util.convert_to_log_fluxes(data_with_spec_z, reduced_flux_keys)

#data_no_spec_z = util.convert_to_log_sn_errors(data_no_spec_z, reduced_flux_keys, reduced_error_keys)
#data_no_spec_z = util.convert_to_log_fluxes(data_no_spec_z, reduced_flux_keys)

keys_in_order = [item for sublist in zip(reduced_flux_keys, reduced_error_keys) for item in sublist]

x = np.asarray(data_with_spec_z[keys_in_order])
y = np.asarray(data_with_spec_z['z_spec']).reshape(-1, 1)

# Split everything into training and validation data sets
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=42)

# Make a network
run_super_name = '18-12-03_cdf_method_improvements_2'
run_name = '12_cdfs=0.30_stds=0.3_m=5_logcosh'

run_dir = './plots/' + run_super_name + '/' + run_name + '/'

try:
    os.mkdir(run_dir)
except FileExistsError:
    print('Not making a new directory because it already exists. I hope you changed the name btw!')

loss_function = loss_funcs.NormalCDFLoss(cdf_strength=0.30, std_deviation_strength=0.3)

network = mdn.MixtureDensityNetwork(loss_function, './logs/' + run_super_name + '/' + run_name,
                                    regularization=None,
                                    x_scaling='standard',
                                    y_scaling=None,
                                    x_features=x_train.shape[-1],
                                    y_features=1,
                                    convolution_layer=True,
                                    convolution_window_size=8,
                                    convolution_stride=4,
                                    layer_sizes=[20, 20, 20],
                                    mixture_components=5,
                                    learning_rate=1e-3)

network.set_training_data(x_train, y_train)

# Run this thing!
exit_code, epochs, training_success = network.train(max_epochs=10000, max_runtime=1.0)

# network.plot_loss_function_evolution()

network.set_validation_data(x_validate, y_validate)

validation_mixtures = network.validate()
validation_results = network.calculate_validation_stats(validation_mixtures)

network.plot_pdf(validation_mixtures, [10, 100, 200, 300, 400, 500],
                 map_values=validation_results['map'],
                 true_values=y_validate.flatten(),
                 figure_directory=run_dir)

z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=True, limits=[0, 7],
                    save_name=run_dir + 'phot_vs_spec.png',
                    plt_title=run_name)

valid_map_values = util.where_valid_redshifts(validation_results['map'])
z_plot.error_evaluator(data_with_spec_z['z_spec'].iloc[valid_map_values], validation_results['map'][valid_map_values],
                       validation_results['lower'][valid_map_values], validation_results['upper'], show_fig=True,
                       save_name=run_dir + 'errors.png',
                       plt_title=run_name)

"""
# Run the pair algorithm on everything that didn't have photometric redshifts
network.set_validation_data(data_no_spec_z[keys_in_order], 0)
validation_mixtures_no_spec_z = network.validate()
validation_results_no_spec_z = network.calculate_validation_stats(validation_mixtures_no_spec_z)

valid_map_values = util.where_valid_redshifts(validation_results_no_spec_z['map'])
all_galaxy_pairs, random_galaxy_pairs = galaxy_pairs.store_pairs_on_sky(data_no_spec_z['ra'].iloc[valid_map_values],
                                                                        data_no_spec_z['dec'].iloc[valid_map_values],
                                                                        max_separation=15., min_separation=1.5,
                                                                        max_move=26, min_move=25,
                                                                        size_of_random_catalogue=1.0,
                                                                        all_pairs_name='all_pairs.csv',
                                                                        random_pairs_name='random_pairs.csv')

max_z = 100.0
min_z = 0.0
all_galaxy_pairs_read_in = galaxy_pairs.read_pairs('./data/all_pairs.csv', validation_results_no_spec_z['map'].iloc[valid_map_values],
                                                   min_redshift=min_z, max_redshift=max_z)

random_galaxy_pairs_read_in = galaxy_pairs.read_pairs('./data/random_pairs.csv', validation_results_no_spec_z['map'].iloc[valid_map_values],
                                                      min_redshift=min_z, max_redshift=max_z)

# Make a plot of Npairs against deltaZ
z_plot.pair_redshift_deviation(validation_results_no_spec_z['map'].iloc[valid_map_values],
                               all_galaxy_pairs_read_in, random_galaxy_pairs_read_in,
                               show_fig=True,
                               save_name='./plots/' + run_super_name + '/' + run_name + '/pairs_all.png',
                               plt_title=run_name + '-- z {:.2f} to {:.2f}'.format(min_z, max_z))


# Initialise twitter
#twit = twitter.TweetWriter()
#twit.write(twitter.initial_text('on 3D-HST data with basic settings.'), reply_to=None)
"""

"""
plt.figure()
plt.plot((validation_results['map'] - y_validate.flatten())/(1 + y_validate.flatten()), mean_sn, 'r.', alpha=0.5)
plt.xlabel(r'$\Delta z / (1 + z_{spec})$')
plt.ylabel('Mean log(S/N)')
plt.gca().invert_yaxis()
plt.xscale('log')
#plt.yscale('log')
plt.title('Mean signal to noise vs redshift error')
#plt.ylim(4, -1)
plt.show()

"""