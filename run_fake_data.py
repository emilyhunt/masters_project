"""Trains a network on fake data from the blog example."""

import numpy as np
import pandas as pd
from scripts import mdn
from scripts import loss_funcs
from scripts import z_plot
from scripts import util
from scripts import twitter


# Import the data
print('Reading in data - bear with, they\'re big files so this may take a while...')
data_train = pd.read_csv('data/galaxy_redshift_sims_train.csv')
data_validation = pd.read_csv('data/galaxy_redshift_sims_valid.csv', nrows=10000)

# Pick out the data we actually want to use
x_train = np.asarray(data_train[['g', 'g_err', 'r', 'r_err', 'i', 'i_err', 'z', 'z_err', 'y', 'y_err']])
y_train = np.asarray(data_train['redshift']).reshape(250000, 1)

x_validate = {}
y_validate = np.asarray(data_validation['redshift']).reshape(10000, 1)
signal_noise_levels = ['SN_1', 'SN_3', 'SN_5']
bands = ['g', 'r', 'i', 'z', 'y']

# Get validation data for each signal to noise level iteratively
for a_signal_noise in signal_noise_levels:

    # Grab keys
    x_keys_to_get = []
    for a_band in bands:
        x_keys_to_get.append(a_band + '_' + a_signal_noise)
        x_keys_to_get.append(a_band + '_err_' + a_signal_noise)

    # Grab the data
    x_validate[a_signal_noise] = np.asarray(data_validation[x_keys_to_get])



# Initialise twitter
#twit = twitter.TweetWriter()
# twit.write(twitter.initial_text('to test small networks and Normal distribution mixtures [running continues EmilyPC + a bugfix x3.]'), reply_to=-1)

# Cycle over a number of different network configurations
rates = [1e-3]#, 5e-3, 5e-4, 1e-4, 1e-3]
sizes = [[20, 20, 10],
        [20, 20, 20],
        [10],
        [10, 10],
        [10, 10, 10]]
regs = [None, 'L1', 'L2']
mixtures = [1, 3, 5]

run_names = []

network = 0

i_sizes = 2
i_regs = 2
i_mixtures = 0

a_reg = None

for a_size in sizes:
    for a_mix in mixtures:
        
        # Decide if we need to use a different loss function
        if a_size[0] is 10:
            a_loss_func = loss_funcs.BetaPDFLoss()
            config_name = str(a_size) + '-' + str(a_mix) + '_' + str(a_reg) + '_'
            func_name = 'beta'
        else:
            a_loss_func = loss_funcs.NormalPDFLoss()
            config_name = str(a_size) + '-' + str(a_mix) + '_' + str(a_reg) + '_norm_'
            func_name = 'normal'
        
        print('==========================================')
        # Setup our network
        del network  # Necessary to stop memory leaks, as re-assigning to network doesn't delete old one properly!
        network = mdn.MixtureDensityNetwork(a_loss_func, './logs/blog_hyperparam_fit/'
                                            + str(a_size) + '_' + str(a_mix) + '_' + str(a_reg) + '_' + func_name,
                                            regularization=a_reg,
                                            x_scaling='robust',
                                            y_scaling='min_max',
                                            x_features=10,
                                            y_features=1,
                                            layer_sizes=a_size,
                                            mixture_components=a_mix,
                                            learning_rate=1e-3)

        network.set_training_data(x_train, y_train)

        # Run this thing!
        exit_code, epochs, training_success = network.train(max_epochs=2000, max_runtime=1.0)

        # network.plot_loss_function_evolution()

        # Calculate how well everything worked, but only if the training was successful
        if training_success:
            validation_mixtures = {}
            validation_results = {}
            points_to_use = 2000
            for a_signal_noise in signal_noise_levels:
                print(a_signal_noise)
                network.set_validation_data(x_validate[a_signal_noise][:points_to_use],
                                            y_validate[:points_to_use].reshape(points_to_use, 1))
                validation_mixtures[a_signal_noise] = network.validate()
                validation_results[a_signal_noise] = network.calculate_validation_stats(validation_mixtures[a_signal_noise],
                                                                                        reporting_interval=int(points_to_use / 5))  # todo: this returns different stuff now after being changed!
                network.plot_pdf(validation_mixtures[a_signal_noise], [10, 100, 200],
                                 map_values=validation_results[a_signal_noise]['map'],
                                 true_values=y_validate[:points_to_use].flatten(),
                                 figure_directory='./plots/18-11-05_blog_hyperparam_tuning/' + config_name + a_signal_noise)

            # Plot all of said results!
            colors = ['r', 'c', 'm']
            for a_signal_noise, a_color in zip(signal_noise_levels, colors):
                z_plot.phot_vs_spec(y_validate[:points_to_use].flatten(), validation_results[a_signal_noise],
                                    save_name='./plots/18-11-05_blog_hyperparam_tuning/zinf_ztrue_' + config_name + a_signal_noise + '.png',
                                    plt_title='Blog data: true vs inferred redshift at ' + a_signal_noise,
                                    point_alpha=0.2, point_color=a_color, limits=[0, 3.0],
                                    show_nmad=True)

"""
        twit.write(twitter.update_text('\nlayer config=' + str(a_size)
                                       + '\nmixtures=' + str(a_mix)
                                       + '\nreg=' + str(a_reg)
                                       + '\nepochs=' + str(epochs)
                                       + '\nlossfunc=' + func_name
                                       + '\nexit code=' + str(exit_code)), reply_to=-1)


twit.write(twitter.annoy_me('Everything is done! YAY'))



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
    validation_redshifts[a_signal_noise] = network.calculate_validation_stats(validation_results[a_signal_noise],
                                                                 reporting_interval=int(points_to_use / 5))

# Plot all of said results!
colors = ['r', 'g', 'c', 'b', 'm']
for a_signal_noise, a_color in zip(signal_noise_levels, colors):
    z_plot.phot_vs_spec(y_validate[:points_to_use], validation_redshifts[a_signal_noise],
                        save_name='./plots/18-11-5_blog_' + a_signal_noise + '.png',
                        plt_title='Blog data: true vs inferred redshift at ' + a_signal_noise,
                        point_alpha=0.2, point_color=a_color, limits=[0, 3.0])

"""
