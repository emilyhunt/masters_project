"""Trains a network on fake data from the blog example."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot


# Import the data
print('Reading in data - bear with, they\'re big files so this may take a while...')
data_train = pd.read_csv('data/galaxy_redshift_sims_train.csv')
data_validation = pd.read_csv('data/galaxy_redshift_sims_valid.csv', nrows=10000)

# Pick out the data we actually want to use
x_train = np.asarray(data_train[['g', 'g_err', 'r', 'r_err', 'i', 'i_err', 'z', 'z_err', 'y', 'y_err']])
y_train = np.asarray(data_train['redshift']).reshape(250000, 1)

x_validate = {}
y_validate = np.asarray(data_validation['redshift']).reshape(10000, 1)
signal_noise_levels = ['SN_1', 'SN_2', 'SN_3', 'SN_4', 'SN_5']
bands = ['g', 'r', 'i', 'z', 'y']

quit()

# Get validation data for each signal to noise level iteratively
for a_signal_noise in signal_noise_levels:

    # Grab keys
    x_keys_to_get = []
    for a_band in bands:
        x_keys_to_get.append(a_band + '_' + a_signal_noise)
        x_keys_to_get.append(a_band + '_err_' + a_signal_noise)

    # Grab the data
    x_validate[a_signal_noise] = np.asarray(data_validation[x_keys_to_get])


# Setup our network
mdn.set_seeds()
network = mdn.MixtureDensityNetwork(loss_funcs.BetaDistribution(), regularization='L2',
                                    x_scaling='robust', y_scaling='min_max', x_features=10, y_features=1,
                                    hidden_layers=3, layer_sizes=[20, 20, 10], mixture_components=5, learning_rate=1e-3)
network.set_training_data(x_train, y_train)

# Run this puppy!
network.train(max_epochs=2000, max_runtime=1.0)
network.plot_loss_function_evolution()

# Calculate how well everything worked!
validation_results = {}
validation_redshifts = {}
points_to_use = 2000
for a_signal_noise in signal_noise_levels:
    print(a_signal_noise)
    network.set_validation_data(x_validate[a_signal_noise][:points_to_use],
                                y_validate[:points_to_use].reshape(points_to_use, 1))
    validation_results[a_signal_noise] = network.validate()
    validation_redshifts[a_signal_noise] = network.calculate_map(validation_results[a_signal_noise],
                                                                 reporting_interval=int(points_to_use / 5))

# Plot all of said results!
colors = ['r', 'g', 'c', 'b', 'm']
for a_signal_noise, a_color in zip(signal_noise_levels, colors):
    z_plot.phot_vs_spec(y_validate[:points_to_use], validation_redshifts[a_signal_noise],
                        save_name='./plots/18-11-2_blog_' + a_signal_noise + '.png',
                        plt_title='Blog data: true vs inferred redshift at ' + a_signal_noise,
                        point_alpha=0.2, point_color=a_color, limits=[0, 3.0])
