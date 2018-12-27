"""Placeholder for a potential future class, depending on how I go about things."""  # todo: docstrings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import pandas as pd
import pymc3
from matplotlib import cm
from scripts import loss_funcs
from scripts.twitter import calc_local_time
from typing import Optional, Union
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize as scipy_minimize
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class MixtureDensityNetwork:

    def __init__(self, loss_function, summary_directory: str, regularization: Optional[str]=None, regularization_scale=0.1,
                 x_features: int=1, y_features: int=1, x_scaling: Optional[str]=None, y_scaling: Optional[str]=None,
                 layer_sizes=15, mixture_components: int=5, learning_rate=0.001, convolution_layer: bool=False,
                 convolution_window_size: int=5, convolution_stride: int=1) -> None:
        """Initialises a mixture density network in tensorflow given the specified (or default) parameters.

        Args:
            loss_function (loss_funcs class): an instance of the desired loss function to use.
            summary_directory (str): location of where we want TensorBoard to write our summaries to.
            regularization (str): controls the type of weight regularisation. Accepts 'none' (default), 'L1' or 'L2'.
            x_features (int): number of input x data points. Default is 1.
            y_features (int): number of input y data points
            layer_sizes (int, list-like): size of one layer (int) or a list of different sizes of each layer.
            mixture_components (int): number of mixtures to try to use todo: fnstring out of date

        Returns:
            None
        """
        # Cast layer_sizes as a 1D numpy array and use it to calculate how many hidden layers there should be
        layer_sizes = np.array([layer_sizes]).flatten()
        hidden_layers = layer_sizes.size

        # Create a tensorflow graph for this class
        with tf.name_scope('graph'):
            self.graph = tf.Graph()
            self.graph_output_names = loss_function.coefficient_names

        # Setup our graph
        with self.graph.as_default():
            # Set seeds for reproducibility
            self.set_seeds()

            # Placeholders for input data
            with tf.variable_scope('data'):

                if convolution_layer:
                    self.x_placeholder = tf.placeholder(tf.float32, [None, x_features, 1], name='x')
                else:
                    self.x_placeholder = tf.placeholder(tf.float32, [None, x_features], name='x')
                self.y_placeholder = tf.placeholder(tf.float32, [None, y_features], name='y')

            # Decide on the type of weight co-efficient regularisation to use based on what the user specified
            with tf.variable_scope('regularization'):
                if regularization is None:
                    self.regularisation_function = None
                    #self.loss_from_regularisation = np.float32(0)
                elif regularization is 'L1':
                    self.regularisation_function = tf.contrib.layers.l1_regularizer(regularization_scale)
                    #self.loss_from_regularisation = tf.losses.get_regularization_loss()
                elif regularization is 'L2':
                    self.regularisation_function = tf.contrib.layers.l2_regularizer(regularization_scale)
                    #self.loss_from_regularisation = tf.losses.get_regularization_loss()
                else:
                    raise ValueError('specified regularisation type is invalid or unsupported.')

            # Setup of the requisite number of layers, kept in a list - this lets us use easy numerical indexes
            # (including just -1 to get to the last one) to access different hidden layers.
            i = 0
            with tf.variable_scope('hidden_layers'):
                self.graph_layers = []

                # Join layers to x data, with a convolution layer if specified
                if convolution_layer:
                    # We 'squeeze' the conv layer to get rid of the last dimension
                    self.graph_layers.append(tf.layers.flatten(
                        tf.layers.conv1d(self.x_placeholder, layer_sizes[i],
                                         [convolution_window_size],
                                         strides=convolution_stride,
                                         activation=tf.nn.tanh,
                                         kernel_regularizer=self.regularisation_function,
                                         name='hidden_layer_1_convolution'),
                        name='convolution_layer_flattening'))
                else:
                    self.graph_layers.append(tf.layers.dense(self.x_placeholder, layer_sizes[i],
                                                             activation=tf.nn.tanh,
                                                             kernel_regularizer=self.regularisation_function,
                                                             name='hidden_layer_1'))

                # Join layers to each other from here on out
                i += 1
                while i < hidden_layers:
                    self.graph_layers.append(tf.layers.dense(self.graph_layers[i - 1], layer_sizes[i],
                                                             activation=tf.nn.tanh,
                                                             kernel_regularizer=self.regularisation_function,
                                                             name='hidden_layer_' + str(i+1)))
                    i += 1

            # Setup of the outputs as a dictionary of output layers, by cycling over the names of outputs and the
            with tf.variable_scope('output_layers'):
                self.graph_output = {}
                for output_name, activation_function, bias_initializer, kernel_initializer in zip(
                        loss_function.coefficient_names, loss_function.activation_functions,
                        loss_function.bias_initializers, loss_function.kernel_initializers):
                    self.graph_output[output_name] = tf.layers.dense(self.graph_layers[-1], mixture_components,
                                                                     activation=activation_function,
                                                                     kernel_regularizer=self.regularisation_function,
                                                                     name='output_' + output_name,
                                                                     bias_initializer=bias_initializer,
                                                                     kernel_initializer=kernel_initializer)

            # Collect all the regularization losses
            with tf.variable_scope('regularization'):
                if regularization is None:
                    self.loss_from_regularisation = np.float32(0)
                else:
                    self.loss_from_regularisation = tf.losses.get_regularization_loss()

            # Initialise the loss function (storing the user-specified one with the class) and training scheme
            with tf.variable_scope('loss_calculation'):
                self.loss_function = loss_function
                loss_function_residuals = self.loss_function.tensor_evaluate(self.y_placeholder, self.graph_output)
                self.loss_from_function = loss_function_residuals['total_residual']

                # Get all the other different residuals from loss_from_function, either using or summary writing them
                residuals = list(loss_function_residuals.keys())
                residuals.remove('total_residual')

                # Record summaries of all the individual residuals
                self.summary_residuals = []
                for a_residual in residuals:
                    self.summary_residuals.append(tf.summary.scalar(a_residual, loss_function_residuals[a_residual]))

                self.loss_total = tf.add(self.loss_from_function, self.loss_from_regularisation)
                self.train_function = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                             name='optimizer').minimize(self.loss_total)

            # Initialise a tensorflow session object using our lovely graph we just made, and initialise the variables
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

            # Initialise some performance evaluation tools
            with tf.variable_scope('summaries'):
                self.summary_loss_function = tf.summary.scalar('loss_function', self.loss_from_function)
                self.summary_loss_regularisation = tf.summary.scalar('loss_regularisation',
                                                                     self.loss_from_regularisation)
                self.summary_loss_total = tf.summary.scalar('loss_total', self.loss_total)

                # Create a summary of the tf.layers variables
                self.summary_of_variables = {}
                for a_var in tf.trainable_variables():
                    self.summary_of_variables[str(a_var)] = tf.summary.histogram(a_var.name, a_var)

                # Make us a cheeky summary writer to write all of our summaries to file
                self.summary_writer = tf.summary.FileWriter(summary_directory, self.graph)

        # Create a blank loss & time array we can append more stuff to later for recording the loss function evolution
        self.recording_steps = np.array([])
        self.loss = np.array([])
        self.epoch_times = np.array([])

        # Initialise blank feed dictionaries and a data range list we use to constrain later MAP value guessing
        self.y_data_range = None
        self.training_data = None
        self.validation_data = None

        # An exit reason to keep class-wide and a training_success bool that we can use to prevent plotting
        self.exit_reason = None
        self.training_success = True

        # Store some deets about the conv layer
        self.convolution_layer = convolution_layer

        # Initialise scalers for the x and y data. We keep these with the class because it means that
        if x_scaling is 'min_max':
            self.x_scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # todo: feature ranges are hard coded!
        elif x_scaling is 'robust':
            self.x_scaler = RobustScaler()
        elif x_scaling is 'standard':
            self.x_scaler = StandardScaler()
        elif x_scaling is None:
            self.x_scaler = None
        else:
            raise NotImplementedError('selected x scaling type has not been implemented!')

        if y_scaling is 'min_max':
            self.y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        elif y_scaling is 'robust':
            self.y_scaler = RobustScaler()
        elif y_scaling is 'standard':
            self.y_scaler = StandardScaler()
        elif y_scaling is None:
            self.y_scaler = None
        else:
            raise NotImplementedError('selected y scaling type has not been implemented!')

        print('An MDN has been initialised!')

    @staticmethod
    def set_seeds(seed=42):
        """Sets seed in numpy and tensorflow. Allows for repeatability!

        Args:
            seed (int, float): universal seed to set in tf and numpy. Default is 42.
        """
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def __del__(self):
        """Closes the tensorflow session."""
        self.session.close()

    def set_training_data(self, x_data, y_data):
        """Modifies the class-unique training feed dictionary to give to the network.

        Args:
            x_data (any): independent variables to feed to the network.
            y_data (any): dependent variables to feed to the network.
        """
        # Scale the x data if requested
        if self.x_scaler is not None:
            x_data = self.x_scaler.fit_transform(x_data)

        # Scale the y data if requested
        if self.y_scaler is not None:
            y_data = self.y_scaler.fit_transform(y_data)

        # Keep an idea of what the maximum and minimum training data range is. This is helpful for plotting later.
        self.y_data_range = [y_data.min(), y_data.max()]

        # Set an extra dimension in the data if we're using a convolution layer
        if self.convolution_layer:
            x_data = np.expand_dims(x_data, axis=2)

        # Add the new x_data, y_data
        self.training_data = {self.x_placeholder: x_data, self.y_placeholder: y_data}

    def set_validation_data(self, x_data, y_data):
        """Modifies the class-unique validation feed dictionary to give to the network.

        Args:
            x_data (any): independent variables to feed to the network.
            y_data (any): dependent variables to feed to the network.
        """
        # Scale the x data if requested, using the same scaling parameters as the training data
        if self.x_scaler is not None:
            x_data = self.x_scaler.transform(x_data)

        # Scale the y data if requested, using the same scaling parameters as the training data
        #if self.y_scaler is not None:
        #    y_data = self.y_scaler.transform(y_data)  todo: does this need ydata??!?!?

        # Set an extra dimension in the data if we're using a convolution layer
        if self.convolution_layer:
            x_data = np.expand_dims(x_data, axis=2)

        # Add the new x_data, y_data
        self.validation_data = {self.x_placeholder: x_data} #, self.y_placeholder: y_data}

    def train(self, max_epochs: int=50, max_runtime: float=1., reporting_time: float=10.,
              max_epochs_without_report: int=1000):
        """Trains the tensorflow graph for the specified amount of time.

        Args:
            max_epochs (int): number of training epochs to run the network for.
            max_runtime (int): maximum number of hours the code will run for before exiting.
            reporting_time (float): how often, in seconds, we print to the console.
            max_epochs_without_report (int): longest time to wait without letting the user know how everything is going
                and storing diagnostic info.

        Returns:
            A list of:
                0. The exit reason
                1. The epoch reached
                2. A bool of whether or not the training was a success
        """
        # optimise: a way to record loss in-place would make this a little bit faster
        # todo: exit automatically when loss function converges.

        # Get a start time and calculate the maximum amount of time we can run for.
        start_time = time.time()
        cutoff_time = time.time() + max_runtime * 60**2

        # Let the user know about our grand plans
        print('\n=== BEGINNING TRAINING ===')
        print('start_time   = {}'.format(calc_local_time(start_time)))
        print('max_epochs   = {}'.format(max_epochs))
        print('max_runtime  = {:.2f} hour'.format(max_runtime))
        print('reporting_time = {:.1f} seconds'.format(reporting_time))
        print('==========================')

        # Correct for if training has been ran before.
        if self.recording_steps.size != 0:
            start_epoch = self.recording_steps[-1]
        else:
            start_epoch = 0
        epoch = start_epoch
        max_epochs = start_epoch + max_epochs

        with self.graph.as_default():
            # Set some defaults so that we don't fail on the first run
            self.exit_reason = 'max_epochs reached'  # Cheeky string that says why we finished training
            have_done_more_than_one_step = False  # Stop stupid predictions being made too early
            epochs_per_report = 1
            step_start_time = time.time()

            # Cycle over, doing some running
            while epoch < max_epochs:

                # Run for requisite number of epochs until refresh (this stops print from being spammed on fast code)
                epochs_in_this_report = 0
                while epochs_in_this_report < epochs_per_report:
                    # optimise: make logging toggleable
                    # Train the network in a new epoch
                    self.session.run(self.train_function, feed_dict=self.training_data)
                    epoch += 1
                    epochs_in_this_report += 1

                # Calculate the new epoch time and ETA
                step_end_time = time.time()
                epoch_time = (step_end_time - step_start_time) / epochs_per_report
                step_start_time = time.time()
                finish_time = (max_epochs - epoch) * epoch_time + step_start_time

                # Record the loss
                self.recording_steps = np.append(self.recording_steps, epoch)
                self.loss = np.append(self.loss, self.session.run(self.loss_from_function, feed_dict=self.training_data))
                self.epoch_times = np.append(self.epoch_times, epoch_time)

                # Check for nan loss, and if not, then write a summary
                if np.isnan(self.loss[-1]):  # todo: could replace this with an exception catch instead for when nan loss happens
                    self.exit_reason = 'nan loss encountered'
                    self.training_success = False
                else:
                    # Write out the summaries
                    summary_merge = tf.summary.merge_all()
                    summary = self.session.run(summary_merge, feed_dict=self.training_data)
                    self.summary_writer.add_summary(summary, epoch + start_epoch)

                # Force garbage collection (sometimes helps to prevent memory issues... sometimes.)
                gc.collect()

                # Work out how many epochs we can do before our next check-in
                epochs_per_report = int(np.ceil(reporting_time / epoch_time))

                # Make sure we aren't gonna go over max_epochs
                if epoch + epochs_per_report > max_epochs:
                    epochs_per_report = max_epochs - epoch
                if epochs_per_report > max_epochs_without_report:
                    epochs_per_report = max_epochs_without_report

                # Output some details on the last few epochs
                print('--------------------------')
                print('CURRENT TIME: {}'.format(calc_local_time(step_start_time)))
                print('epoch       = {} ({:.1f}% done)'.format(epoch, epoch / float(max_epochs) * 100))
                print('epoch_time  = {:.3f} seconds'.format(epoch_time))
                print('loss        = {:.5f}'.format(self.loss[-1]))

                if have_done_more_than_one_step:
                    print('finish_time = {}'.format(calc_local_time(finish_time)))
                    # todo: evaluate whether or not we're gonna finish before the cutoff time

                # Decide if we need to end
                if step_start_time > cutoff_time:
                    self.exit_reason = 'time limit reached'
                    self.training_success = True
                    break
                if self.exit_reason is 'nan loss encountered':
                    self.training_success = False
                    break

                have_done_more_than_one_step = True

        # Make sure to update the training success bool if it did indeed work
        if self.exit_reason == 'max_epochs reached':
            self.training_success = True

        # It's all over! =( (we let the user know, unsurprisingly)
        print('=== ENDING TRAINING ======')
        print('reason      = {}'.format(self.exit_reason))
        print('epochs done = {}'.format(epoch))
        print('==========================')

        return [self.exit_reason, epoch, self.training_success]

    def save_graph(self, location: str):
        """Saves a complete copy of the current network to a specified location.
        Args:
            location (str): place where you want it saved, including the filename.
        Returns:
            None
        """
        # Make a tf.train.Saver object and use it to save the graph.
        with self.graph.as_default():
            saver = tf.train.Saver()
            save_path = saver.save(self.session, location)

        print("A copy of the model has been saved to {}".format(save_path))

    def open_graph(self, location: str):
        """Saves a complete copy of the current network to a specified location.
        Args:
            location (str): place where you want it saved, including the filename.
        Returns:
            None
        """
        # todo: make this work, as it requires doing some extra stuff with loss functions and writing shit to the class
        pass

    def validate(self):
        """Returns mixture parameters for the code given the verification data.

        Returns:
            Validation data mixture co-efficients in a dictionary that sorts them by name.
        """
        print('Validating the graph on the validation data...')
        # Run the session on validation data with the aim of returning mixture co-efficients
        result = {}

        # Cycle over the different output constants, writing their validation run to 'result'
        for a_constant in self.graph_output_names:
            result[a_constant] = self.session.run(self.graph_output[a_constant], feed_dict=self.validation_data)

        return result

    def plot_loss_function_evolution(self, start: int=0, end: int=-1, y_log: bool=False,
                                     figure_directory: Optional=None) -> None:
        """Returns a plot of the change of the loss function over time.

        Args:
            start (int): start epoch to plot.
            end (int): end epoch to plot. Default: -1, which sets the end to the last training step.
            y_log (bool): if True, sets y axis to be logarithmic.
        """
        print('Plotting the evolution of the loss function...')

        # See if the user wants the end to be the most recent loss function evaluation.
        if end == -1:
            end = self.loss.size

        # Plot some stuff!
        fig = plt.figure(figsize=(4, 6))
        ax_loss = fig.add_subplot(2, 1, 1)
        ax_time = fig.add_subplot(2, 1, 2, sharex=ax_loss)
        ax_loss.plot(self.recording_steps, self.loss[start:end], 'r-', label='loss')
        ax_time.plot(self.recording_steps, self.epoch_times[start:end], 'b-', label='epoch time')

        ax_loss.set_title('Loss function evolution')
        ax_loss.set_ylabel('Loss')

        ax_time.set_xlabel('Epoch')
        ax_time.set_ylabel('Epoch time')

        # Set the plot to be log if desired
        if y_log:
            plt.yscale('log')

        # Save if desired
        if figure_directory is not None:
            plt.savefig(figure_directory + '_lossf.png')

        plt.show()

    def calculate_validation_stats(self, validation_mixtures, data_range: Union[tuple, list], reporting_interval: int=100,
                                   start_resolution: int=100,
                                   uncertainty_integration_resolution: int=2000, uncertainty_sigma_level: float=1.):
        """Calculates the MAP (maximum a posteriori), uncertainty and modality of a given set of mixture distributions.

        Args:
            validation_mixtures (dict): the data from a .validate call.
            data_range (list-like of length 2): the allowed range of data.
            reporting_interval (int): how often to let the user know which objects we're working on. Default: 100.
            start_resolution (int): number of points to test against when finding the initial guess.
            uncertainty_integration_resolution (int): number of random variables to draw when integrating the pdfs to
                find the uncertainty levels.
            uncertainty_sigma_level (float): todo: implement this! calculate with erf(x / sqrt(2))

        Returns:
            A pandas.DataFrame() containing map values, limits and a pdf modality count, addressed with keys 'map',
            'lower', 'upper' and 'modality'.

        """
        # Only do this next step if prior training actually worked!
        if self.training_success is False:
            print('Prior training failed! Unable to calculate MAP values. Exiting calculate_MAP.')
            return pd.DataFrame()

        print('Attempting to calculate the MAP values of all distributions...')

        # Define a function to minimise, multiplied by -1 to make sure we're minimising not maximising
        def function_to_minimise(x_data, my_object_dictionary, my_loss_function):
            return -1 * my_loss_function.pdf_single_point(x_data, my_object_dictionary)

        # Create blank arrays of np.nan values to populate with hopefully successful minimisations
        n_objects = validation_mixtures[self.graph_output_names[0]].shape[0]
        map_values = np.empty(n_objects)
        map_values[:] = np.nan
        upper_limits = map_values.copy()
        lower_limits = map_values.copy()
        cdf_constant = map_values.copy()
        cdf_multiplier = map_values.copy()
        modality = np.zeros(n_objects, dtype=int)
        valid_map_values = np.zeros(n_objects, dtype=int)

        # A bit of setup for our initial guess of MAP values
        guess_x_range = np.linspace(data_range[0], data_range[1], num=start_resolution)

        # Loop over each object and work out the stats for each
        i = 0
        successes = 0
        mean_number_of_iterations = 0.0
        while i < n_objects:

            # Make a dictionary that only has data on this specific galaxy
            object_dictionary = {}
            for a_name in self.graph_output_names:
                object_dictionary[a_name] = validation_mixtures[a_name][i]

            # Make a sensible starting guess and look at the y values in parameter space
            guess_y_range = self.loss_function.pdf_multiple_points(guess_x_range, object_dictionary, sum_mixtures=True)
            starting_guess = guess_x_range[np.argmax(guess_y_range)]

            # Estimate the modality of the distribution by looking for sign changes, using a method from:
            # https://stackoverflow.com/questions/2652368/how-to-detect-a-sign-change-for-elements-in-a-numpy-array
            # First, we find the difference between consecutive values and hence estimate the gradient
            guess_difference = guess_y_range[1:] - guess_y_range[0:-1]

            # Then, we find the sign of the gradient (np.sign returns +1 or -1) and then look for signchanges, where
            # consecutive guess_sign values being added or subtracted make +2 or -2 (not zero.) We also only care about
            # sign changes that occur for points that have a y value at least 1% of the MAP (as we don't care about
            # tiny modes or accidentally detecting the flat tails either side of the distribution.)
            guess_sign = np.sign(guess_difference)
            guess_signchange = np.logical_and(guess_sign[1:] - guess_sign[0:-1] != 0,
                                              guess_y_range[1:-1] > guess_y_range.max() * 0.01)

            # Count the number of turning points in the pdf
            turning_points = np.count_nonzero(guess_signchange)

            # Convert this number of turning points into a measure of modality with some fancyness
            if turning_points != 0:
                modality[i] = int(np.ceil(0.5 + turning_points / 2))
            else:
                modality[i] = 0

            # Attempt to minimise and find the MAP value
            result = scipy_minimize(function_to_minimise, np.array([starting_guess]),
                                    args=(object_dictionary, self.loss_function), method='Nelder-Mead',
                                    options={'disp': False})

            # Store the result only if we're able to
            if result.success:
                map_values[i] = result.x
                valid_map_values[i] = 1
                successes += 1
                mean_number_of_iterations += result.nit

                # Now, use random deviates to find the uncertainty on the MAP
                random_deviates = self.loss_function.draw_random_variables(uncertainty_integration_resolution,
                                                                           object_dictionary)

                # pymc3 docs for the following functions: https://docs.pymc.io/api/stats.html
                # Use the highest probability density region if the pdf is unimodal
                if modality[i] == 1:
                    limits = pymc3.stats.hpd(random_deviates, alpha=0.3173)  # Corresponds to 1 sigma of error.
                    lower_limits[i] = limits[0]
                    upper_limits[i] = limits[1]

                # If not, then we find quantiles corresponding to +- 1 sigma. This isn't theoretically as
                # good, as the IQR is not guaranteed to contain the MAP value we found earlier.
                else:
                    limits = pymc3.stats.quantiles(random_deviates, qlist=[15.865, 84.135])  # Again, 1 sigma error.
                    lower_limits[i] = limits[15.865]
                    upper_limits[i] = limits[84.135]

                    # Set the MAP to the median if the MAP we found isn't within the limits
                    if map_values[i] > upper_limits[i] or map_values[i] < lower_limits[i]:
                        map_values[i] = np.median(random_deviates)
                        valid_map_values[i] = 2

                # Calculate constants to let us normalise the CDFs later
                cdf_stuff = self.loss_function.cdf_multiple_points(data_range, object_dictionary)
                cdf_constant[i] = -1. * cdf_stuff[0]
                cdf_multiplier[i] = 1. / cdf_stuff[1]

            else:
                print('Failed to find MAP for object {}!'.format(i))

            # Keep the user updated on what interval of objects we're working on (prevents panic if this takes ages)
            if i % reporting_interval == 0:
                print('Working on objects {} to {}...'.format(i, i+reporting_interval))

            i += 1

        # Scale the data in reverse if we're using a y-data scaler
        if self.y_scaler is not None:
            finite_map_values = np.isfinite(map_values)
            map_values[finite_map_values] = self.y_scaler.inverse_transform(map_values[finite_map_values]
                                                                            .reshape(-1, 1)).flatten()
            finite_lower_limits = np.isfinite(lower_limits)
            lower_limits[finite_lower_limits] = self.y_scaler.inverse_transform(lower_limits[finite_lower_limits]
                                                                                .reshape(-1, 1)).flatten()
            finite_upper_limits = np.isfinite(upper_limits)
            upper_limits[finite_upper_limits] = self.y_scaler.inverse_transform(upper_limits[finite_upper_limits]
                                                                                .reshape(-1, 1)).flatten()

        # Calculate the mean number of iterations of the minimizer
        mean_number_of_iterations /= float(successes)

        print('Found MAP values for {:.2f}% of objects.'.format(100 * successes / float(n_objects)))
        print('Mean number of MAP minimisation iterations = {}'.format(mean_number_of_iterations))

        return pd.DataFrame({'map': map_values, 'lower': lower_limits, 'upper': upper_limits, 'modality': modality,
                             'valid': valid_map_values, 'cdf_constant': cdf_constant, 'cdf_multiplier': cdf_multiplier})

    def plot_pdf(self, validation_data: dict, values_to_highlight, data_range=None, resolution: int=100,
                 map_values=None, true_values=None, figure_directory: Optional[str]=None, show_fig: bool=False):
        """Plots the mixture pdf of a given set of parameters.

        # todo: true_values should get flattened first as there can be issues with it being an odd shape

        Args:
            validation_data (dict): as returned by network.validate, this is the validation data to plot with.
            values_to_highlight (int, list-like of ints): IDs of the objects to plot pdfs for. Default: None.
            data_range (list-like of floats): The range
            resolution (int): how many points to evaluate the pdf at.
            map_values ( #todo FINISH YOUR FUCKING DOCSTRINGS EMILY FFS

        Returns:
            pretty graphs
        """
        # Only do this next step if prior training actually worked!
        if self.training_success is False:
            print('Prior training failed! Unable to plot PDFs. Exiting calculate_MAP.')
            return 0

        # See if the user specified their own data range
        if data_range is None:
            data_range = self.y_data_range

        # Typecast the list of values to highlight as a numpy 1D array
        values_to_highlight = np.array([values_to_highlight]).flatten()
        n_mixtures = validation_data[self.graph_output_names[0]][0, :].size

        # Cycle over intrinsic_start and intrinsic_end. We scale back into actual value space later.
        y_range = np.linspace(data_range[0], data_range[1], num=resolution)
        if self.y_scaler is not None:
            actual_y_range = self.y_scaler.inverse_transform(y_range.reshape(-1, 1)).flatten()
        else:
            actual_y_range = y_range

        # Setup a list of colours to plot each mixture with
        colors = cm.viridis(np.linspace(0, 1, n_mixtures))

        # Evaluate all the different pdfs.
        for an_object in values_to_highlight:

            # Make a dictionary that only has data on this specific galaxy
            object_dictionary = {}
            for a_name in self.graph_output_names:
                object_dictionary[a_name] = validation_data[a_name][an_object]

            # Get the pdf data, including a total probability
            mixture_pdfs = self.loss_function.pdf_multiple_points(y_range, object_dictionary, sum_mixtures=False)
            total_pdf = np.sum(mixture_pdfs, axis=0)

            # Make some cool plots
            plt.figure()

            # Plot the total mixture
            plt.plot(actual_y_range, total_pdf, 'k-', lw=2, label='Total pdf', alpha=1)

            # Plot each mixture individually
            for mixture_number, a_color in enumerate(colors):
                plt.plot(actual_y_range, mixture_pdfs[mixture_number, :], '--', lw=1, color=a_color,
                         label='Mixture ' + str(mixture_number))

            # If specified, plot the MAP value
            if map_values is not None:
                plt.plot([map_values[an_object], map_values[an_object]], [total_pdf.min(), total_pdf.max()],
                         'r--', label='MAP value')

            # If specified, plot the MAP value
            if true_values is not None:
                plt.plot([true_values[an_object], true_values[an_object]], [total_pdf.min(), total_pdf.max()],
                         'b--', label='True value')

            plt.legend(edgecolor='k', facecolor='w', fancybox=True)
            plt.title('PDF of object ' + str(an_object))

            # Save if desired
            if figure_directory is not None:
                plt.savefig(figure_directory + '_pdf_' + str(an_object) + '.png')

            if show_fig:
                plt.show()
            else:
                plt.close()
                # todo make this stuff work


# Unit tests: implements the class on the toy_mdn_emily example data, using data from the following blog post:
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
if __name__ == '__main__':
    print('Commencing mdn.py unit tests!')

    # Create some data to play wit1h
    def build_toy_dataset(dataset_size):
        y_data = np.random.uniform(5, 10.5, dataset_size)  # DEFAULT: -10.5, +10.5
        r_data = np.random.normal(size=dataset_size)  # random noise
        x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 0.1
        x_data = x_data.reshape((dataset_size, 1))
        y_data = y_data.reshape((dataset_size, 1))
        return train_test_split(x_data, y_data, random_state=42)

    points = 5000

    x_train, x_test, y_train, y_test = build_toy_dataset(points)
    print("Size of features in training data: {}".format(x_train.shape))
    print("Size of output in training data: {}".format(y_train.shape))
    print("Size of features in test data: {}".format(x_test.shape))
    print("Size of output in test data: {}".format(y_test.shape))
    # plt.plot(x_train, y_train, 'or', mew=0, ms=3, alpha=0.2)
    # plt.title('Training data')
    # plt.show()

    # Initialise the network
    network = MixtureDensityNetwork(loss_funcs.NormalPDFLoss(),
                                    './logs/mdn_tests_tensorboard/' + str(time.strftime('%H-%M-%S', time.localtime(time.time()))),
                                    regularization=None,
                                    x_features=1, y_features=1, layer_sizes=[20, 20],
                                    mixture_components=15, learning_rate=5e-3,
                                    y_scaling=None, x_scaling=None)

    # Set the data
    network.set_training_data(x_train, y_train)
    network.set_validation_data(x_test, y_test)

    # Train the network for max_epochs epochs
    network.train(max_epochs=250)

    # Plot the loss function
    network.plot_loss_function_evolution()

    # Validate the network
    validation_results = network.validate()

    # Calculate some MAP values
    validation_stats = network.calculate_validation_stats(validation_results, reporting_interval=500)

    # Plot some pdfs
    network.plot_pdf(validation_results, [0, 100, 200], map_values=validation_stats['map'], show_fig=True)


    # Some code to plot a validation plot. Firstly, we have to generate points to pull from:
    def generate_points(my_x_test, my_weights, my_means, my_std_deviations):
        """Generates points randomly given a loada points. Uses uniform deviates to guess a mixture coefficient to use.
        Then draws a point randomly from said selected distribution. We do this instead of picking the mode because
        we're fitting a model to data with intrinsic scatter!"""
        n_test_points = my_x_test.size

        # Pick out some uniform deviates
        mixtures_to_use = np.random.rand(n_test_points).reshape(n_test_points, 1)

        # Create a cumulative sum of weights
        my_weights_sum = np.cumsum(my_weights, axis=1)

        # Find the first argument that's greater than the uniform deviate (since np.argmax stops at the first instance)
        random_weights_indexes = np.argmax(np.greater(my_weights_sum, mixtures_to_use), axis=1)

        # Grab the random means and standard deviations
        random_means = my_means[np.arange(0, n_test_points), random_weights_indexes]
        random_std_deviations = my_std_deviations[np.arange(0, n_test_points), random_weights_indexes]

        # Use these parameters to make some random numbers that are normal distributed
        return np.random.normal(loc=random_means, scale=random_std_deviations)

    # Make some points
    y_test_random = generate_points(x_test,
                                    validation_results[network.graph_output_names[0]],
                                    validation_results[network.graph_output_names[1]],
                                    validation_results[network.graph_output_names[2]])

    # Plot some stuff
    plt.figure()
    plt.plot(x_test, y_test, 'or', mew=0, ms=3, alpha=0.5, label='Training data')
    plt.plot(x_test, y_test_random, 'ob', mew=0, ms=3, alpha=0.5, label='Predictions')
    plt.title('Network prediction vs training data')
    plt.legend(fancybox=True)
    plt.ylim(-25, 25)
    plt.show()
