"""Placeholder for a potential future class, depending on how I go about things."""  # todo: docstrings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scripts import loss_funcs
from sklearn.model_selection import train_test_split


def calc_local_time(input_time):
    """Convenience function to convert any times to prettier time strings.

    Args:
        input_time (float): the time.time() in seconds you wish to convert to a nice string.
    """
    return time.strftime('%c', time.localtime(input_time))


def set_seeds(seed=42):
    """Sets seed in numpy and tensorflow. Allows for repeatability!

    Args:
        seed (int, float): universal seed to set in tf and numpy. Default is 42.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


class MixtureDensityNetwork:

    def __init__(self, loss_function, regularization: str='none', regularization_scale=0.1, x_features: int=1,
                 y_features: int=1, hidden_layers: int=2, layer_sizes=15, mixture_components: int=5) -> None:
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
            self.x_placeholder = tf.placeholder(tf.float32, [None, x_features])
            self.y_placeholder = tf.placeholder(tf.float32, [None, y_features])

            # Decide on the type of weight co-efficient regularisation to use based on what the user specified
            if regularization is 'none':
                self.regularisation_function = None
                self.regularisation_loss = 0
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
            for output_name, activation_function in zip(loss_function.coefficient_names,
                                                        loss_function.activation_functions):
                self.graph_output[output_name] = tf.layers.dense(self.graph_layers[-1], mixture_components,
                                                                 activation=activation_function,
                                                                 kernel_regularizer=self.regularisation_function)

            # Initialise the loss function and training scheme
            self.loss_function = tf.add(loss_function.evaluate(self.y_placeholder, self.graph_output),
                                        self.regularisation_loss)
            self.train_function = tf.train.AdamOptimizer().minimize(self.loss_function)

            # Initialise a tensorflow session object using our lovely graph we just made, and initialise the variables
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        # Create a blank loss array we can append more stuff to later for recording the loss function evolution
        self.loss = np.array([])

        # Initialise blank feed dictionaries
        self.training_data = None
        self.validation_data = None

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
        # Add the new x_data, y_data
        self.training_data = {self.x_placeholder: x_data, self.y_placeholder: y_data}

    def set_validation_data(self, x_data, y_data):
        """Modifies the class-unique validation feed dictionary to give to the network.

        Args:
            x_data (any): independent variables to feed to the network.
            y_data (any): dependent variables to feed to the network.
        """
        # Add the new x_data, y_data
        self.validation_data = {self.x_placeholder: x_data, self.y_placeholder: y_data}

    def train(self, max_epochs: int=50, max_runtime: float=1., reporting_time: float=10) -> None:
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
            self.loss[epoch + start_epoch] = self.session.run(self.loss_function, feed_dict=self.training_data)
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
                    self.loss[epoch + start_epoch] = self.session.run(self.loss_function, feed_dict=self.training_data)
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

    def calculate_map(self):  # todo
        """Calculates the MAP (maximum a posteriori) of a given set of mixture distributions"""
        pass

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
    network = MixtureDensityNetwork(loss_funcs.NormalDistribution(), regularization='L2',
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

