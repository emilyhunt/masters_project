"""Trains a network on the real 3dhst good south data."""
import numpy as np
import os
import pandas as pd
import time

import scripts.file_handling
from scripts import preprocessing
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scipy.optimize import minimize as scipy_minimize
from scripts import galaxy_pairs
from scripts import twitter
from data import hst_goodss_config
from sklearn.model_selection import train_test_split


# Begin by reading in the data
data = scripts.file_handling.read_fits('./data/goodss_3dhst.v4.1.cat.FITS', hst_goodss_config.keys_to_keep,
                                       new_column_names=None, get_flux_and_error_keys=False)

archival_redshifts = scripts.file_handling.read_save('./data/KMOS415_output/GS415.3dhst.redshift.save')
archival_colours = scripts.file_handling.read_save('./data/KMOS415_output/GS415.rf.save')
archival_sfr = scripts.file_handling.read_save('./data/KMOS415_output/GS415.SFR.save')
archival_lmass = scripts.file_handling.read_save('./data/KMOS415_output/GS415.zbest.all.ltaugt8.5.bc03.ch.save')

data[['z_spec', 'z_phot_lit', 'z_phot_lit_l68', 'z_phot_lit_u68', 'z_grism', 'z_grism_l68', 'z_grism_u68']] = \
    archival_redshifts[['gs4_zspec', 'gs4_zphot', 'gs4_zphot_l68', 'gs4_zphot_u68', 'gs4_zgrism', 'gs4_zgrism_l68',
                        'gs4_zgrism_u68']]

data[['rf_u', 'rf_b', 'rf_v', 'rf_r', 'rf_j']] = \
    archival_colours[['gs4_rfu', 'gs4_rfb', 'gs4_rfv', 'gs4_rfr', 'gs4_rfj']]

data[['sfr']] = archival_sfr[['gs4_sfrbest']]

data[['sed_log_sfr', 'sed_log_mass', 'sed_metal', 'sed_log_age']] = \
    archival_lmass[['gs4_sed_lsfr', 'gs4_sed_lmass', 'gs4_sed_metal', 'gs4_sed_lage']]


# Get some useful things from the config file
flux_keys = hst_goodss_config.flux_keys_in_order
error_keys = hst_goodss_config.error_keys_in_order
band_central_wavelengths = hst_goodss_config.band_central_wavelengths.copy()

# Take a look at the coverage in different photometric bands
data_with_spec_z, data_no_spec_z, reduced_flux_keys, reduced_error_keys = \
    scripts.preprocessing.missing_data_handler(data, flux_keys, error_keys, band_central_wavelengths,
                                               coverage_minimum=0.99,
                                               valid_photometry_column='use_phot',
                                               missing_flux_handling='normalised_column_mean_ratio',
                                               missing_error_handling='5_sigma_column')

# RANDOM RESCALING
#data_with_spec_z[reduced_flux_keys + reduced_error_keys] =

# Initialise a PhotometryScaler to use to log scale etc everything in the same way
preprocessor = preprocessing.PhotometryScaler([data_with_spec_z, data_no_spec_z],
                                              reduced_flux_keys, reduced_error_keys)

# Split everything into training and validation data sets
data_training, data_validation = preprocessor.train_validation_split(data_with_spec_z, training_set_size=0.75, seed=42)

# Make some extra data to use in a moment
data_training = preprocessor.enlarge_dataset_within_error(data_training, reduced_flux_keys, reduced_error_keys,
                                                          min_sn=0.0, max_sn=2.0, error_model='exponential',
                                                          error_correlation='row-wise', outlier_model=None,
                                                          dataset_scaling_method='random', edsd_mean_redshift=1.2,
                                                          new_dataset_size_factor=5., clip_fluxes=False)

#data_validation = preprocessor.enlarge_dataset_within_error(data_validation, reduced_flux_keys, reduced_error_keys,
#                                                          min_sn=0.0, max_sn=1.0, error_model='uniform',
#                                                          error_correlation='row-wise', outlier_model=None,
#                                                          new_dataset_size_factor=10.)

# Convert everything that matters to log fluxes
data_training = preprocessor.convert_to_log_sn_errors(data_training, reduced_flux_keys, reduced_error_keys)
data_training = preprocessor.convert_to_zeroed_magnitudes(data_training, reduced_flux_keys)

data_validation = preprocessor.convert_to_log_sn_errors(data_validation, reduced_flux_keys, reduced_error_keys)
data_validation = preprocessor.convert_to_zeroed_magnitudes(data_validation, reduced_flux_keys)

data_no_spec_z = preprocessor.convert_to_log_sn_errors(data_no_spec_z, reduced_flux_keys, reduced_error_keys)
data_no_spec_z = preprocessor.convert_to_zeroed_magnitudes(data_no_spec_z, reduced_flux_keys)


# Grab keys in order and make final training/validation arrays
keys_in_order = [item for sublist in zip(reduced_flux_keys, reduced_error_keys) for item in sublist]  # ty StackExchange
x_train = data_training[keys_in_order].values
y_train = data_training['z_spec'].values.reshape(-1, 1)
x_validate = data_validation[keys_in_order].values
y_validate = data_validation['z_spec'].values.reshape(-1, 1)

# Make a network
run_super_name = '18-12-22_pc_optimisation'
run_name = '46_fully_automated_speed_communism'

run_dir = './plots/' + run_super_name + '/' + run_name + '/'  # Note: os.makedirs() won't accept pardirs like '..'

try:
    os.mkdir(run_dir)
except FileExistsError:
    print('Not making a new directory because it already exists. I hope you changed the name btw!')


def function_to_minimise(x):
    loss_function = loss_funcs.NormalPDFLoss(perturbation_coefficient_0=x[0],
                                             perturbation_coefficient_1=0.0,
                                             perturbation_coefficient_2=0.0,
                                             perturbation_coefficient_3=0.0)

    network = mdn.MixtureDensityNetwork(loss_function, './logs/' + run_super_name + '/' + run_name,
                                        regularization=None,
                                        regularization_scale=0.1,
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
    exit_code, epochs, training_success = network.train(max_epochs=2000, max_runtime=2.0, max_epochs_without_report=500)

    # Validate the network at different signal to noise mutliplier levels
    if training_success is False:
        return np.inf

    network.set_validation_data(x_validate, y_validate)

    validation_mixtures = network.validate()
    validation_results = network.calculate_validation_stats(validation_mixtures, [0, 7])

    overall_nmad = z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=False,
                                       limits=[0, 7])

    mean_residual, max_residual = z_plot.error_evaluator_wittman(y_validate.flatten(),
                                                                 validation_mixtures, validation_results,
                                                                 show_fig=False)

    print(x)
    print(overall_nmad)
    print(max_residual)

    return np.log(overall_nmad) + np.log(max_residual)


results1 = scipy_minimize(function_to_minimise, np.array([0.0207]),
                         method='Nelder-Mead',
                         options={'disp': True})

print(results1.x)
