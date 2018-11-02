"""Placeholder for a potential future class, depending on how I go about things."""  # todo: docstrings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from matplotlib import cm
from scripts import loss_funcs
from typing import Optional
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize as scipy_minimize
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def set_seeds(seed=42):
    """Sets seed in numpy and tensorflow. Allows for repeatability!

    Args:
        seed (int, float): universal seed to set in tf and numpy. Default is 42.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def calc_local_time(input_time):
    """Convenience function to convert any times to prettier time strings.

    Args:
        input_time (float): the time.time() in seconds you wish to convert to a nice string.
    """
    return time.strftime('%c', time.localtime(input_time))


class MixtureDensityNetwork:

    def __init__(self, loss_function, regularization: Optional[str]=None, regularization_scale=0.1,
                 x_features: int=1, y_features: int=1, x_scaling: Optional[str]=None, y_scaling: Optional[str]=None,
                 hidden_layers: int=2, layer_sizes=15, mixture_components: int=5, learning_rate=0.001) -> None:
        """Initialises a mixture density network in tensorflow given the specified (or default) parameters.

        Args:
            loss_function (loss_funcs class): an instance of the desired loss function to use.
            regularization (str): controls the type of weight regularisation. Accepts 'none' (default), 'L1' or 'L2'.
            x_features (int): number of input x data points. Default is 1.
            y_features (int): number of input y data points
            hidden_layers (int): number of hidden layers
            layer_sizes (int, list-like): sizes of all layers (int) or a list of different sizes of each layer.
            mixture_components (int): number of mixtures to try to use

        Returns:
            None
        """
        # Work out whether or not layer_sizes is a list of layer sizes or an integer. If it's just an integer,
        # then make it into an array of the same integer repeated.
        if isinstance(layer_sizes, int):
            layer_sizes = np.ones(hidden_layers, dtype=int) * layer_sizes

        # Create a tensorflow graph for this class
        self.graph = tf.Graph()
        self.graph_output_names = loss_function.coefficient_names

        # Setup our graph
        with self.graph.as_default():
            # Placeholders for input data
            self.x_placeholder = tf.placeholder(tf.float64, [None, x_features])
            self.y_placeholder = tf.placeholder(tf.float64, [None, y_features])

            # Decide on the type of weight co-efficient regularisation to use based on what the user specified
            if regularization is None:
                self.regularisation_function = None
                self.regularisation_loss = np.float64(0)
            elif regularization is 'L1':
                self.regularisation_function = tf.contrib.layers.l1_regularizer(regularization_scale)
                self.regularisation_loss = tf.losses.get_regularization_loss()
            elif regularization is 'L2':
                self.regularisation_function = tf.contrib.layers.l2_regularizer(regularization_scale)
                self.regularisation_loss = tf.losses.get_regularization_loss()
            else:
                raise ValueError('specified regularisation type is invalid or unsupported.')

            # Setup of the requisite number of layers, kept in a list - this lets us use easy numerical indexes
            # (including just -1 to get to the last one) to access different hidden layers.
            i = 0
            self.graph_layers = []

            # Join layers to x data
            self.graph_layers.append(tf.layers.dense(self.x_placeholder, layer_sizes[i], activation=tf.nn.relu,
                                                     kernel_regularizer=self.regularisation_function))

            # Join layers to each other from here on out
            i += 1
            while i < hidden_layers:
                self.graph_layers.append(tf.layers.dense(self.graph_layers[i - 1], layer_sizes[i],
                                                         activation=tf.nn.relu,
                                                         kernel_regularizer=self.regularisation_function))
                i += 1

            # Setup of the outputs as a dictionary of output layers, by cycling over the names of outputs and the
            self.graph_output = {}
            for output_name, activation_function, bias_initializer, kernel_initializer in zip(
                    loss_function.coefficient_names, loss_function.activation_functions,
                    loss_function.bias_initializers, loss_function.kernel_initializers):
                self.graph_output[output_name] = tf.layers.dense(self.graph_layers[-1], mixture_components,
                                                                 activation=activation_function,
                                                                 kernel_regularizer=self.regularisation_function,
                                                                 bias_initializer=bias_initializer,
                                                                 kernel_initializer=kernel_initializer)

            # Initialise the loss function (storing the user-specified one with the class) and training scheme
            self.loss_function = loss_function
            self.loss_function_tensor = tf.add(self.loss_function.tensor_evaluate(self.y_placeholder,
                                                                                  self.graph_output),
                                               self.regularisation_loss)
            self.train_function = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_function_tensor)

            # Initialise a tensorflow session object using our lovely graph we just made, and initialise the variables
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        # Create a blank loss array we can append more stuff to later for recording the loss function evolution
        self.loss = np.array([])

        # Initialise blank feed dictionaries and a data range list we use to constrain later MAP value guessing
        self.y_data_range = None
        self.training_data = None
        self.validation_data = None

        # Initialise scalers for the x and y data. We keep these with the class because it means that
        if x_scaling is 'min_max':
            self.x_scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # todo: feature ranges are hard coded!
        elif x_scaling is 'robust':
            self.x_scaler = RobustScaler()
        elif x_scaling is None:
            self.x_scaler = None
        else:
            raise NotImplementedError('selected x scaling type has not been implemented!')

        if y_scaling is 'min_max':
            self.y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        elif y_scaling is 'robust':
            self.y_scaler = RobustScaler()
        elif y_scaling is None:
            self.y_scaler = None
        else:
            raise NotImplementedError('selected y scaling type has not been implemented!')

        print('An MDN has been initialised!')

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

        self.y_data_range = [y_data.min(), y_data.max()]

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
        if self.y_scaler is not None:
            y_data = self.y_scaler.transform(y_data)

        # Add the new x_data, y_data
        self.validation_data = {self.x_placeholder: x_data, self.y_placeholder: y_data}

    def train(self, max_epochs: int=50, max_runtime: float=1., reporting_time: float=10.) -> None:
        """Trains the tensorflow graph for the specified amount of time.

        Args:
            max_epochs (int): number of training epochs to run the network for.
            max_runtime (int): maximum number of hours the code will run for before exiting.
            reporting_time (float): how often, in seconds, we print to the console.
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

        # Correct for if training has been ran before. Note that epoch is one-indexed, so we subtract one from
        # start_epoch to make sure calls to self.loss are all correct.
        start_epoch = self.loss.size - 1
        epoch = 1

        # Append blank space onto the loss evolution recording array
        self.loss = np.append(self.loss, np.empty(max_epochs))

        with self.graph.as_default():
            # Time the first step to get an estimate of how long each step will take
            epoch_time = time.time()
            self.session.run(self.train_function, feed_dict=self.training_data)
            self.loss[epoch + start_epoch] = self.session.run(self.loss_function_tensor, feed_dict=self.training_data)
            epoch_time = time.time() - epoch_time

            # Cycle over, doing some running
            exit_reason = 'max_epochs reached'  # Cheeky string that says why we finished training
            have_done_more_than_one_step = False  # Stop stupid predictions being made too early
            while epoch < max_epochs:
                step_start_time = time.time()

                # Work out how many epochs we can do before our next check-in
                epochs_per_report = int(np.ceil(reporting_time / epoch_time))

                # Make sure we aren't gonna go over max_epochs
                if epoch + epochs_per_report > max_epochs:
                    epochs_per_report = max_epochs - epoch

                # Run for requisite number of epochs until refresh (this stops print from being spammed on fast code)
                epochs_in_this_report = 0
                while epochs_in_this_report < epochs_per_report:
                    epoch += 1
                    self.session.run(self.train_function, feed_dict=self.training_data)
                    self.loss[epoch + start_epoch] = self.session.run(self.loss_function_tensor,
                                                                      feed_dict=self.training_data)
                    epochs_in_this_report += 1

                # Calculate the new epoch time and ETA
                now_time = time.time()
                finish_time = (max_epochs - epoch) * epoch_time + now_time

                # Output some details on the last few epochs
                print('--------------------------')
                print('CURRENT TIME: {}'.format(calc_local_time(now_time)))
                print('epoch       = {} ({:.1f}% done)'.format(epoch, epoch / float(max_epochs) * 100))
                print('epoch_time  = {:.3f} seconds'.format(epoch_time))
                print('loss        = {:.5f}'.format(self.loss[epoch + start_epoch - 1]))

                if have_done_more_than_one_step:
                    print('finish_time = {}'.format(calc_local_time(finish_time)))

                # Decide if we need to end
                if now_time > cutoff_time:
                    exit_reason = 'time limit reached'

                have_done_more_than_one_step = True
                step_end_time = time.time()
                epoch_time = (step_end_time - step_start_time) / epochs_per_report

        # It's all over! =( (we let the user know, unsurprisingly)
        print('=== ENDING TRAINING ======')
        print('reason      = {}'.format(exit_reason))
        print('epochs done = {}'.format(epoch))
        print('==========================')

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

    def plot_loss_function_evolution(self, start: int=0, end: int=-1, y_log: bool=False) -> None:
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
        plt.figure()
        plt.plot(np.arange(start, end), self.loss[start:end], 'r-')
        plt.title('Loss function evolution')

        # Set the plot to be log if desired, and show
        if y_log:
            plt.yscale('log')
        plt.show()

    def calculate_map(self, validation_data, reporting_interval: int=100):
        """Calculates the MAP (maximum a posteriori) of a given set of mixture distributions."""
        print('Attempting to calculate the MAP values of all distributions...')

        # Define a function to minimise, multiplied by -1 to make sure we're minimising not maximising
        def function_to_minimise(x_data, my_object_dictionary, my_loss_function):
            return -1 * my_loss_function.pdf_single_point(x_data, my_object_dictionary)

        # Create a blank array of np.nan values to populate with hopefully successful minimisations
        n_objects = validation_data[self.graph_output_names[0]].shape[0]
        map_values = np.empty(n_objects)

        # A bit of setup for our initial guess of MAP values
        guess_x_range = np.linspace(self.y_data_range[0], self.y_data_range[1], num=20)

        # Loop over each object and work out the MAP values for each
        i = 0
        successes = 0
        mean_number_of_iterations = 0.0
        while i < n_objects:

            # Make a dictionary that only has data on this specific galaxy
            object_dictionary = {}
            for a_name in self.graph_output_names:
                object_dictionary[a_name] = validation_data[a_name][i]

            # Make a sensible starting guess
            starting_guess = guess_x_range[np.argmax(self.loss_function.pdf_multiple_points(guess_x_range,
                object_dictionary, sum_mixtures=True))]

            # Attempt to minimise and find the MAP value
            result = scipy_minimize(function_to_minimise, np.array([starting_guess]),
                                    args=(object_dictionary, self.loss_function), method='Nelder-Mead',
                                    options={'disp': False})

            # Store the result only if we're able to
            if result.success:
                map_values[i] = result.x
                successes += 1
                mean_number_of_iterations += result.nit
            else:
                print('Failed to find MAP for object {}!'.format(i))
                map_values[i] = np.nan

            # Keep the user updated on what interval of objects we're working on (prevents panic if this takes ages)
            if i % reporting_interval == 0:
                print('Working on objects {} to {}...'.format(i, i+reporting_interval))

            i += 1

        # Scale the data in reverse if we're using a y-data scaler
        if self.y_scaler is not None:
            finite_map_values = np.isfinite(map_values)
            map_values[finite_map_values] = self.y_scaler.inverse_transform(map_values[finite_map_values]
                                                                            .reshape(-1, 1)).flatten()

        # Calculate the mean number of iterations of the minimizer
        mean_number_of_iterations /= float(successes)

        print('Found MAP values for {:.2f}% of objects.'.format(100 * successes / float(n_objects)))
        print('Mean number of iterations = {}'.format(mean_number_of_iterations))
        return map_values

    def plot_pdf(self, validation_data: dict, values_to_highlight, intrinsic_start: float=0., intrinsic_end: float=1.,
                 resolution: int=100):
        """Plots the mixture pdf of a given set of parameters.

        Args:
            validation_data (dict): as returned by network.validate, this is the validation data to plot with.
            values_to_highlight (int, list-like of ints): IDs of the objects to plot pdfs for
            intrinsic_start (float): start value of the pdf, in loss function space. Needs to be set sensibly based on
                                     the properties of the data.
            intrinsic_end (float): as above, but the end.
            resolution (int): how many points to evaluate the pdf at.

        Returns:
            pretty graphs
        """
        # Typecast the list of values to highlight as a numpy 1D array
        values_to_highlight = np.array([values_to_highlight]).flatten()
        n_mixtures = validation_data[self.graph_output_names[0]][0, :].size

        # Cycle over intrinsic_start and intrinsic_end. We scale back into actual value space later.
        y_range = np.linspace(intrinsic_start, intrinsic_end, num=resolution)
        actual_y_range = self.y_scaler.inverse_transform(y_range.reshape(-1, 1)).flatten()

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

            # Plot each mixture individually
            for mixture_number, a_color in enumerate(colors):
                plt.plot(actual_y_range, mixture_pdfs[mixture_number, :], '-', color=a_color,
                         label='Mixture ' + str(mixture_number))

            # Plot the total mixture
            plt.plot(actual_y_range, total_pdf, 'k-', lw=4, label='Total pdf')
            plt.legend(edgecolor='k', facecolor='w', fancybox=True)
            plt.title('PDF of object ' + str(an_object))
            plt.show()

    def calculate_5050(self):  # todo
        """Calculates the central mean of a given set of mixture distributions"""
        pass


# Unit tests: implements the class on the toy_mdn_emily example data, using data from the following blog post:
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
if __name__ == '__main__':
    print('Commencing mdn.py unit tests!')

    # Set the seed
    set_seeds()

    # Create some data to play with
    def build_toy_dataset(dataset_size):
        y_data = np.random.uniform(-10.5, 10.5, dataset_size)
        r_data = np.random.normal(size=dataset_size)  # random noise
        x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
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
    network = MixtureDensityNetwork(loss_funcs.NormalDistribution(), regularization=None,
                                    x_features=1, y_features=1, hidden_layers=2, layer_sizes=[25, 15],
                                    mixture_components=15)

    # Set the data
    network.set_training_data(x_train, y_train)
    network.set_validation_data(x_test, y_test)

    # Train the network for max_epochs epochs
    network.train(max_epochs=3000)

    # Plot the loss function
    network.plot_loss_function_evolution()

    # Validate the network
    validation_results = network.validate()

    # Calculate some MAP values
    map_values = network.calculate_map(validation_results, reporting_interval=500)

    # Plot some pdfs
    network.plot_pdf(validation_results, [0, 1, 2, 3], intrinsic_start=0, intrinsic_end=1)


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

