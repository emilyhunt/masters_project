"""Trains a network on fake data from the blog example."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from scripts import mdn
from scripts import loss_funcs


# Import the data
print('Reading in data - bear with, they\'re big files so this may take a while...')
data_train = pd.read_csv('data/galaxy_redshift_sims_train.csv')
data_validation = pd.read_csv('data/galaxy_redshift_sims_valid.csv')

# Pick out the data we actually want to use and scale it
print('Scaling data to required ranges and properties...')
robust_scaler = RobustScaler()
min_max_scaler = MinMaxScaler(feature_range=(0.0001, 0.9999))

x_train = robust_scaler.fit_transform(np.asarray(data_train[['g', 'g_err', 'r', 'r_err', 'i', 'i_err', 'z', 'z_err', 'y',
                                                       'y_err']]))
y_train = min_max_scaler.fit_transform(np.asarray(data_train['redshift']).reshape(250000, 1))

x_validate = {}
y_validate = min_max_scaler.transform(np.asarray(data_validation['redshift']).reshape(1378950, 1))
signal_noise_levels = ['SN_1', 'SN_2', 'SN_3', 'SN_4', 'SN_5']
bands = ['g', 'r', 'i', 'z', 'y']

# Get validation data for each signal to noise level iteratively
for a_signal_noise in signal_noise_levels:

    # Grab keys
    x_keys_to_get = []
    for a_band in bands:
        x_keys_to_get.append(a_band + '_' + a_signal_noise)
        x_keys_to_get.append(a_band + '_err_' + a_signal_noise)

    # Grab the data
    x_validate[a_signal_noise] = robust_scaler.transform(np.asarray(data_validation[x_keys_to_get]))


# Setup our network
mdn.set_seeds()
network = mdn.MixtureDensityNetwork(loss_funcs.BetaDistribution(), regularization='none',
                                    x_features=10, y_features=1, hidden_layers=2, layer_sizes=[20, 20],
                                    mixture_components=5)
network.set_training_data(x_train, y_train)

# Run this puppy!
network.train(max_epochs=2000, max_runtime=0.5)
network.plot_loss_function_evolution()
