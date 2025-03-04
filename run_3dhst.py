"""Trains a network on the real 3dhst good south data."""

import os
import scripts.file_handling
import numpy as np
from scripts import preprocessing
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scripts import galaxy_pairs
from data import hst_goodss_config


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
                                                          new_dataset_size_factor=20., clip_fluxes=False)

#data_validation = preprocessor.enlarge_dataset_within_error(data_validation, reduced_flux_keys, reduced_error_keys,
#                                                          min_sn=0.0, max_sn=1.0, error_model='uniform',
#                                                          error_correlation='row-wise', outlier_model=None,
#                                                          new_dataset_size_factor=10.)

# Convert everything that matters to log fluxes
data_training = preprocessor.convert_to_log_sn_errors(data_training, reduced_flux_keys, reduced_error_keys)
data_training = preprocessor.convert_to_zeroed_magnitudes(data_training, reduced_flux_keys)

data_no_spec_z = preprocessor.convert_to_log_sn_errors(data_no_spec_z, reduced_flux_keys, reduced_error_keys)
data_no_spec_z = preprocessor.convert_to_zeroed_magnitudes(data_no_spec_z, reduced_flux_keys)

# Grab keys in order and make final training/validation arrays
keys_in_order = [item for sublist in zip(reduced_flux_keys, reduced_error_keys) for item in sublist]  # ty StackExchange
x_train = data_training[keys_in_order].values
y_train = data_training['z_spec'].values.reshape(-1, 1)

# Make a network
run_super_name = '18-12-27_very_long_runs'
run_name = '14_intentionally_shit_data'

run_dir = './plots/' + run_super_name + '/' + run_name + '/'  # Note: os.makedirs() won't accept pardirs like '..'

try:
    os.mkdir(run_dir)
except FileExistsError:
    print('Not making a new directory because it already exists. I hope you changed the name btw!')

#loss_function = loss_funcs.NormalCDFLoss(cdf_strength=0.00, std_deviation_strength=0.0, normalisation_strength=1.0,
#                                         grid_size=100, mixtures=5)

loss_function = loss_funcs.NormalPDFLoss(perturbation_coefficient_0=0.028)

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
exit_code, epochs, training_success = network.train(max_epochs=40000, max_runtime=5.0,
                                                    max_epochs_without_report=5000, reporting_time=120.)

# Validate the network at different signal to noise mutliplier levels
sn_multipliers = [4., 3., 2., 1., 20., 0.]

for a_sn in sn_multipliers:
    data_validation_temp = preprocessor.enlarge_dataset_within_error(data_validation, reduced_flux_keys, reduced_error_keys,
                                                                min_sn=a_sn, max_sn=a_sn, error_model='uniform',
                                                                error_correlation='row-wise', outlier_model=None,
                                                                new_dataset_size_factor=None, clip_fluxes=False)

    a_sn = '_sn=' + str(a_sn)

    data_validation_temp = preprocessor.convert_to_log_sn_errors(data_validation_temp, reduced_flux_keys, reduced_error_keys)
    data_validation_temp = preprocessor.convert_to_zeroed_magnitudes(data_validation_temp, reduced_flux_keys)

    x_validate = data_validation_temp[keys_in_order].values
    y_validate = data_validation_temp['z_spec'].values.reshape(-1, 1)

    network.set_validation_data(x_validate, y_validate)

    validation_mixtures = network.validate()
    validation_results = network.calculate_validation_stats(validation_mixtures, [0, 7])

    network.plot_pdf(validation_mixtures, [10, 100, 200, 300, 400, 500],
                     map_values=validation_results['map'],
                     true_values=y_validate.flatten(),
                     figure_directory=run_dir + a_sn)

    overall_nmad = z_plot.phot_vs_spec(y_validate.flatten(), validation_results['map'], show_nmad=True, show_fig=False,
                                       limits=[0, 7],
                                       save_name=run_dir + 'phot_vs_spec' + a_sn + '.png',
                                       plt_title=None)

    valid_map_values = util.where_valid_redshifts(validation_results['map'])
    z_plot.error_evaluator(y_validate.flatten(), validation_results['map'],
                           validation_results['lower'], validation_results['upper'], show_fig=False,
                           save_name=run_dir + 'errors' + a_sn + '.png',
                           plt_title=None)

    mean_residual, max_residual = z_plot.error_evaluator_wittman(y_validate.flatten(),
                                                                 validation_mixtures, validation_results, show_fig=False,
                                                                 save_name=run_dir + 'wittman_errors' + a_sn + '.png',
                                                                 plt_title=None)

    z_plot.population_histogram(validation_results['map'], bins='auto', color='m',
                                plt_title='Validation data ML redshift distribution',
                                show_fig=False,
                                save_name=run_dir + 'population_histogram_ML_valid' + a_sn + '.png')

    # Save stuff to csv!
    to_csv = [data_validation, validation_results]
    to_name = ['data_validation' + a_sn, 'validation_results' + a_sn]
    for name, csv_me in zip(to_name, to_csv):
        csv_me.to_csv('./final_run_data/different_sn/' + name + '.csv')
        print('Successfully saved {}'.format(name))

    # Pickle some other stuff!
    import pickle

    to_pickle = [validation_mixtures]
    to_name = ['validation_mixtures' + a_sn]
    for name, pickle_me in zip(to_name, to_pickle):
        with open('./final_run_data/different_sn/' + name + '.pkl', 'wb') as f:
            pickle.dump(pickle_me, f, pickle.HIGHEST_PROTOCOL)
        print('Successfully saved {}'.format(name))

    # NB to open again:
    # with open('name.pkl', 'rb') as f:
    #     thing = pickle.load(f)




# Run the pair algorithm on everything that didn't have photometric redshifts
network.set_validation_data(data_no_spec_z[keys_in_order], 0)
validation_mixtures_no_spec_z = network.validate()
validation_results_no_spec_z = network.calculate_validation_stats(validation_mixtures_no_spec_z,  [0, 7],
                                                                  reporting_interval=1000)

overall_nmad = z_plot.phot_vs_spec(data_no_spec_z['z_phot_lit'], validation_results_no_spec_z['map'], show_nmad=True,
                                   show_fig=False,
                                   limits=[0, 7],
                                   save_name=run_dir + 'phot_vs_EAZY.png',
                                   plt_title=None, point_alpha=0.02, point_color='b')

overall_nmad = z_plot.phot_vs_spec(data_no_spec_z['z_grism'], validation_results_no_spec_z['map'], show_nmad=True,
                                   show_fig=False,
                                   limits=[0, 7],
                                   save_name=run_dir + 'phot_vs_grism.png',
                                   plt_title=None, point_alpha=0.02, point_color='b')

valid_map_values = util.where_valid_redshifts(validation_results_no_spec_z['map'])
all_galaxy_pairs, random_galaxy_pairs = galaxy_pairs.store_pairs_on_sky(data_no_spec_z['ra'].iloc[valid_map_values],
                                                                        data_no_spec_z['dec'].iloc[valid_map_values],
                                                                        max_separation=15., min_separation=1.5,
                                                                        max_move=26, min_move=25,
                                                                        size_of_random_catalogue=1.0,
                                                                        all_pairs_name='all_pairs.csv',
                                                                        random_pairs_name='random_pairs.csv')

network.plot_pdf(validation_mixtures_no_spec_z, np.arange(0, 20),
                 map_values=validation_results_no_spec_z['map'],
                 true_values=data_no_spec_z['z_phot_lit'],
                 figure_directory=run_dir + 'test_dataset')

network.plot_pdf(validation_mixtures_no_spec_z, np.arange(10000, 10020),
                 map_values=validation_results_no_spec_z['map'],
                 true_values=data_no_spec_z['z_phot_lit'],
                 figure_directory=run_dir + 'test_dataset')

network.plot_pdf(validation_mixtures_no_spec_z, np.arange(20000, 20020),
                 map_values=validation_results_no_spec_z['map'],
                 true_values=data_no_spec_z['z_phot_lit'],
                 figure_directory=run_dir + 'test_dataset')

network.plot_pdf(validation_mixtures_no_spec_z, np.arange(30000, 30020),
                 map_values=validation_results_no_spec_z['map'],
                 true_values=data_no_spec_z['z_phot_lit'],
                 figure_directory=run_dir + 'test_dataset')

max_z = 100.0
min_z = 0.0
all_galaxy_pairs_read_in = galaxy_pairs.read_pairs('./data/all_pairs.csv', validation_results_no_spec_z['map'].iloc[valid_map_values],
                                                   min_redshift=min_z, max_redshift=max_z)

random_galaxy_pairs_read_in = galaxy_pairs.read_pairs('./data/random_pairs.csv', validation_results_no_spec_z['map'].iloc[valid_map_values],
                                                      min_redshift=min_z, max_redshift=max_z)

# Make a plot of Npairs against deltaZ
z_plot.pair_redshift_deviation(validation_results_no_spec_z['map'].iloc[valid_map_values],
                               all_galaxy_pairs_read_in, random_galaxy_pairs_read_in,
                               show_fig=False,
                               save_name='./plots/' + run_super_name + '/' + run_name + '/pairs_all.png',
                               plt_title=None)

"""

z_plot.population_histogram(data_no_spec_z['z_phot_lit'], bins='auto', color='b',
                            plt_title='Test data EAZY redshift distribution',
                            show_fig=False,
                            save_name=run_dir + 'population_histogram_EAZY_test')

z_plot.population_histogram(validation_results_no_spec_z['map'], bins='auto', color='r',
                            plt_title='Test data ML redshift distribution',
                            show_fig=False,
                            save_name=run_dir + 'population_histogram_ML_test')


import matplotlib.pyplot as plt

# Colour plot
y = data_no_spec_z['rf_u'] - data_no_spec_z['rf_v']
x = data_no_spec_z['rf_v'] - data_no_spec_z['rf_j']

val = 0.8
test = (np.abs(validation_results_no_spec_z['map'] - data_no_spec_z['z_phot_lit'])
        / (1 + 0.5*(validation_results_no_spec_z['map'] + data_no_spec_z['z_phot_lit'])))
good = np.where(test < val)[0]
bad = np.where(test > val)[0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x[good], y[good], 'k.', ms=1, alpha=0.05)
ax.plot(np.array([]), np.array([]), 'k.', ms=1, alpha=0.3, label=r'$\Delta z / (1 + z_{mean}) < $' + str(val))
ax.plot(x[bad], y[bad], 'rs', ms=1, alpha=0.05, label=r'$\Delta z / (1 + z_{mean}) > $' + str(val))
ax.set_ylabel('U-V')
ax.set_xlabel('V-J')
ax.set_title('Location of outliers using EAZY rest frame colours')
ax.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)
fig.show()


# H band magnitude plot
y = data_no_spec_z['rf_u'] - data_no_spec_z['rf_v']
x = data_no_spec_z['rf_v'] - data_no_spec_z['rf_j']

val = 0.8
test = (np.abs(validation_results_no_spec_z['map'] - data_no_spec_z['z_phot_lit'])
        / (1 + 0.5*(validation_results_no_spec_z['map'] + data_no_spec_z['z_phot_lit'])))
good = np.where(test < val)[0]
bad = np.where(test > val)[0]

f = data_no_spec_z['f_f160w']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(f[:], test[:], 'k.', ms=1, alpha=0.05)
#ax.plot(np.array([]), np.array([]), 'k.', ms=1, alpha=0.3, label=r'$\Delta z / (1 + z_{mean}) < $' + str(val))
#ax.plot(f[bad], test[bad], 'rs', ms=1, alpha=0.3, label=r'$\Delta z / (1 + z_{mean}) > $' + str(val))
ax.set_ylabel('$\Delta z / (1 + z_{mean})$')
ax.set_xlabel('H-band magnitude')
ax.set_title('Location of outliers using EAZY rest frame colours')
ax.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)
fig.show()


# Band and outlier plot
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.hist(data_no_spec_z['rf_j'], bins='auto', color='r', alpha=0.5, label='Test data')
ax.set_ylabel('Frequency - test')
#ax.set_xlabel('H band magnitude')
ax.set_title('Dataset H band magnitude distribution')
ax.hist(data_training['rf_j'], bins='auto', color='b', alpha=0.5, label='Training data')
ax.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)
#ax.set_xlim(-30, 0)

test = (np.abs(validation_results_no_spec_z['map'] - data_no_spec_z['z_phot_lit'])
        / (1 + 0.5*(validation_results_no_spec_z['map'] + data_no_spec_z['z_phot_lit'])))
ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
ax2.plot(data_no_spec_z['rf_j'], test, 'k.', ms=1, alpha=0.05)
ax2.set_ylabel(r'$\Delta z / (1 + z_{mean})$')
ax2.set_xlabel('H band magnitude')
#ax2.set_yscale('log')
#ax2.set_xlim(-30, 0)

fig.show()

test = (np.abs(validation_results_no_spec_z['map'] - data_no_spec_z['z_phot_lit'])
        / (1 + 0.5*(validation_results_no_spec_z['map'] + data_no_spec_z['z_phot_lit'])))
z_plot.density_plot(data_no_spec_z['sed_log_mass'], data_no_spec_z['sed_log_sfr'], test, show_fig=True, x_lim=(0, 12), y_lim=(-13, 4),
                    x_label=r'$log(M_{stars})$', y_label=r'$log(SFR)$', grid_resolution=15, n_levels=20,
                    plt_title='SFR vs log stellar mass: test dataset',
                    save_name=run_dir + 'sfr_vs_mstars_test.png')

test = (np.abs(validation_results['map'] - data_validation['z_spec'])
        / (1 + 0.5*(validation_results['map'] + data_validation['z_spec'])))
z_plot.density_plot(data_validation['sed_log_mass'], data_validation['sed_log_sfr'], test, show_fig=True, x_lim=(0, 12), y_lim=(-13, 4),
                    x_label=r'$log(M_{stars})$', y_label=r'$log(SFR)$', grid_resolution=15, n_levels=20,
                    plt_title='SFR vs log stellar mass: validation dataset', point_alpha=0.2,
                    save_name=run_dir + 'sfr_vs_mstars_valid.png')
"""
"""
# Save stuff to csv!
to_csv = [data_training, data_validation, data_no_spec_z, validation_results, validation_results_no_spec_z]
to_name = ['data_training', 'data_validation', 'data_no_spec_z', 'validation_results', 'validation_results_no_spec_z']
for name, csv_me in zip(to_name, to_csv):
    csv_me.to_csv('./final_run_data/' + name + '.csv')
    print('Successfully saved {}'.format(name))

# Pickle some other stuff!
import pickle
to_pickle = [validation_mixtures, validation_mixtures_no_spec_z]
to_name = ['validation_mixtures', 'validation_mixtures_no_spec_z']
for name, pickle_me in zip(to_name, to_pickle):
    with open('./final_run_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(pickle_me, f, pickle.HIGHEST_PROTOCOL)
    print('Successfully saved {}'.format(name))

# NB to open again:
# with open('name.pkl', 'wb') as f:
#     thing = pickle.load(f)
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