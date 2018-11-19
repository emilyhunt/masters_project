"""Trains a network on the real 3dhst good south data."""

import numpy as np
import pandas as pd
import time
from astropy.io import fits
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scripts import galaxy_pairs
from scripts import twitter
from data import hst_goodss_config
from sklearn.model_selection import train_test_split


# Begin by reading in the data
fits_file = fits.open('./data/goodss_3dhst.v4.1.cat.FITS')

data, flux_keys, error_keys = util.make_3dhst_photometry_table(fits_file[1], hst_goodss_config.keys_to_keep,
                                                               new_key_names=None)

archival_redshifts = util.read_save('./data/KMOS415_output/GS415.3dhst.redshift.save')

data[['z_spec', 'z_phot_lit', 'z_phot_lit_l68', 'z_phot_lit_u68']] = \
    archival_redshifts[['gs4_zspec', 'gs4_zphot', 'gs4_zphot_l68', 'gs4_zphot_u68']]


# Take a look at the coverage in different photometric bands
columns_to_train_with, columns_to_remove, rows_to_remove_bad_phot, rows_to_remove_incomplete_phot = \
    util.check_photometric_coverage(data, flux_keys + error_keys, coverage_minimum=0.60,
                                    valid_photometry_column='use_phot')

"""Returned on 19/11/18 with coverage_minimum at 0.6:
I have checked the coverage of the data. I found that:
10509 of 50507 rows had a bad photometry warning flag and are not included.
4 out of 80 columns do not have coverage over 60.0% on good sources.
These were: ['f_f606wcand', 'f_h', 'e_f606wcand', 'e_h']
I also found that 15831 of 39998 rows would still have invalid values even after removing all the above.
This leaves 47.85% of rows in the final data set.
"""

# Be reaaaally harsh about removing lots of poor data
data = data.drop(columns=columns_to_remove,
                 index=np.append(rows_to_remove_bad_phot, rows_to_remove_incomplete_phot)).reset_index()

has_spec_z = np.where(data['z_spec'] != -99.0)[0]

x = np.asarray(data[columns_to_train_with].iloc[has_spec_z])
y = np.asarray(data['z_spec'].iloc[has_spec_z]).reshape(-1, 1)

# Split everything into training and validation data sets
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=42)

# Make a network
loss_function = loss_funcs.BetaDistribution()

network = mdn.MixtureDensityNetwork(loss_function, './logs/candels_run_1/'
                                    + str(time.strftime('%H-%M-%S', time.localtime(time.time()))),
                                    regularization=None,
                                    x_scaling='robust',
                                    y_scaling='min_max',
                                    x_features=x_train.shape[1],
                                    y_features=1,
                                    layer_sizes=[20, 20, 10],
                                    mixture_components=5,
                                    learning_rate=1e-3)

network.set_training_data(x_train, y_train)

# Run this thing!
exit_code, epochs, training_success = network.train(max_epochs=10, max_runtime=1.0)

network.plot_loss_function_evolution()

network.set_validation_data(x_validate, y_validate)

validation_mixtures = network.validate()
validation_results = network.calculate_validation_stats(validation_mixtures)

network.plot_pdf(validation_results, [10, 100, 200],
                 map_values=validation_results['map'],
                 true_values=y_validate.flatten(),
                 figure_directory='./plots/18-11-19_candels_run_1/')

z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=True, limits=[0, 7],
                    save_name='./plots/18-11-19_candels_run_1/phot_vs_spec.png')

z_plot.sky_locations(data['ra'].iloc[has_spec_z], data['dec'].iloc[has_spec_z])


"""
all_galaxy_pairs, random_galaxy_pairs = galaxy_pairs.store_pairs_on_sky(data['ra'].iloc[has_spec_z],
                                                                        data['dec'].iloc[has_spec_z],
                                                                        max_separation=15., min_separation=1.5,
                                                                        max_move=26, min_move=25,
                                                                        size_of_random_catalogue=1.0,
                                                                        all_pairs_name='candels_1_all_pairs.csv',
                                                                        random_pairs_name='candels_1_random_pairs.csv')

max_z = 100.0
    min_z = 0.0
    all_galaxy_pairs_read_in = scripts.galaxy_pairs.read_pairs('./data/candels_1_all_pairs.csv', redshifts['gs4_zphot'],
                                                               min_redshift=min_z, max_redshift=max_z)

    random_galaxy_pairs_read_in = scripts.galaxy_pairs.read_pairs('./data/candels_1_random_pairs.csv', redshifts['gs4_zphot'],
                                                                  min_redshift=min_z, max_redshift=max_z,
                                                                  size_of_random_catalogue=random_catalogue_repeats)

    # Make a plot of Npairs against deltaZ
    pair_redshift_deviation(redshifts['gs4_zphot'], all_galaxy_pairs_read_in, random_galaxy_pairs_read_in,
                            size_of_random_catalogue=random_catalogue_repeats)










# Initialise twitter
#twit = twitter.TweetWriter()
#twit.write(twitter.initial_text('on 3D-HST data with basic settings.'), reply_to=None)
"""