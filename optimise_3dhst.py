"""Trains a network on the real 3dhst good south data."""

import numpy as np
import os

import scripts.file_handling
import scripts.preprocessing
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scipy.optimize import minimize as scipy_minimize
from data import hst_goodss_config
from sklearn.model_selection import train_test_split


# Begin by reading in the data
data = scripts.file_handling.read_fits('./data/goodss_3dhst.v4.1.cat.FITS', hst_goodss_config.keys_to_keep,
                                       new_column_names=None, get_flux_and_error_keys=False)

archival_redshifts = scripts.file_handling.read_save('./data/KMOS415_output/GS415.3dhst.redshift.save')

data[['z_spec', 'z_phot_lit', 'z_phot_lit_l68', 'z_phot_lit_u68']] = \
    archival_redshifts[['gs4_zspec', 'gs4_zphot', 'gs4_zphot_l68', 'gs4_zphot_u68']]

# Get some useful things from the config file
flux_keys = hst_goodss_config.flux_keys_in_order
error_keys = hst_goodss_config.error_keys_in_order
band_central_wavelengths = hst_goodss_config.band_central_wavelengths

# Take a look at the coverage in different photometric bands
data_with_spec_z, data_no_spec_z, reduced_flux_keys, reduced_error_keys = \
    scripts.preprocessing.missing_data_handler(data, flux_keys, error_keys, band_central_wavelengths,
                                               coverage_minimum=0.5,
                                               valid_photometry_column='use_phot',
                                               missing_flux_handling='normalised_column_mean',
                                               missing_error_handling='big_value')

data_with_spec_z = scripts.preprocessing.convert_to_log_sn_errors(data_with_spec_z, reduced_flux_keys, reduced_error_keys)
data_with_spec_z = scripts.preprocessing.convert_to_log_fluxes(data_with_spec_z, reduced_flux_keys)

#data_no_spec_z = util.convert_to_log_sn_errors(data_no_spec_z, reduced_flux_keys, reduced_error_keys)
#data_no_spec_z = util.convert_to_log_fluxes(data_no_spec_z, reduced_flux_keys)

keys_in_order = [item for sublist in zip(reduced_flux_keys, reduced_error_keys) for item in sublist]

x = np.asarray(data_with_spec_z[keys_in_order])
y = np.asarray(data_with_spec_z['z_spec']).reshape(-1, 1)

# Split everything into training and validation data sets
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=42)

# Make a network
run_super_name = '18-12-03_cdf_method_improvements_2'
run_name = '18_parameters_optimised'

run_dir = './plots/' + run_super_name + '/' + run_name + '/'

try:
    os.mkdir(run_dir)
except FileExistsError:
    print('Not making a new directory because it already exists. I hope you changed the name btw!')


def function_to_minimise(x):

    cdf_strength = x[0]
    std_deviation_strength = x[1]
    loss_function = loss_funcs.NormalCDFLoss(cdf_strength=cdf_strength, std_deviation_strength=std_deviation_strength,
                                             grid_size=100, mixtures=3)

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
                                        mixture_components=3,
                                        learning_rate=1e-3)

    network.set_training_data(x_train, y_train)

    # Run this thing!
    exit_code, epochs, training_success = network.train(max_epochs=3000, max_runtime=1.0, reporting_time=300.)

    if training_success is False:
        return np.inf

    # network.plot_loss_function_evolution()

    network.set_validation_data(x_validate, y_validate)

    validation_mixtures = network.validate()
    validation_results = network.calculate_validation_stats(validation_mixtures)

    overall_nmad = z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=False,
                                       limits=[0, 7],
                                       save_name=None,
                                       plt_title=run_name)

    mean_residual, max_residual = z_plot.error_evaluator_wittman(y_validate.flatten(),
                                                                 validation_mixtures, show_fig=False,
                                                                 save_name=None,
                                                                 plt_title=run_name)

    print("cdf_strength: {}".format(cdf_strength))
    print("std_deviation_strength: {}".format(std_deviation_strength))
    print("nmad: {}".format(overall_nmad))
    print("mean_residaul: {}".format(mean_residual))
    print("max_residual: {}".format(max_residual))

    return overall_nmad * max_residual


results1 = scipy_minimize(function_to_minimise, np.array([0.60, 0.15]),
                         method='Nelder-Mead',
                         options={'disp': True})

print(results1.x)


def function_to_minimise(x):
    cdf_strength = x[0]
    std_deviation_strength = x[1]
    loss_function = loss_funcs.NormalCDFLoss(cdf_strength=cdf_strength, std_deviation_strength=std_deviation_strength,
                                             grid_size=100, mixtures=3)

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
                                        mixture_components=3,
                                        learning_rate=1e-3)

    network.set_training_data(x_train, y_train)

    # Run this thing!
    exit_code, epochs, training_success = network.train(max_epochs=3000, max_runtime=1.0, reporting_time=300.)

    if training_success is False:
        return np.inf

    # network.plot_loss_function_evolution()

    network.set_validation_data(x_validate, y_validate)

    validation_mixtures = network.validate()
    validation_results = network.calculate_validation_stats(validation_mixtures)

    overall_nmad = z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=False,
                                       limits=[0, 7],
                                       save_name=None,
                                       plt_title=run_name)

    mean_residual, max_residual = z_plot.error_evaluator_wittman(y_validate.flatten(),
                                                                 validation_mixtures, show_fig=False,
                                                                 save_name=None,
                                                                 plt_title=run_name)

    print("cdf_strength: {}".format(cdf_strength))
    print("std_deviation_strength: {}".format(std_deviation_strength))
    print("nmad: {}".format(overall_nmad))
    print("mean_residaul: {}".format(mean_residual))
    print("max_residual: {}".format(max_residual))

    return overall_nmad * mean_residual


results2 = scipy_minimize(function_to_minimise, np.array([0.60, 0.15]),
                          method='Nelder-Mead',
                          options={'disp': True})

print(results1.x)
print(results2.x)
