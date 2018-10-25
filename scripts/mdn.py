"""Placeholder for a potential future class, depending on how I go about things."""  # todo: docstrings

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts import loss_funcs


class MixtureDensityNetwork:

    def __init__(self, loss_function: str, x_features: int=1, y_features: int=1, hidden_layers: int=2,
                 layer_sizes=15, mixture_components: int=5) -> None:
        """Initialises a mixture density network in tensorflow given the specified (or default) parameters.

        Args:
            loss_function (str): name of the desired loss function
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

        # Setup our graph
        with self.graph.as_default():
            # Placeholders for input data
            self.x_placeholder = tf.placeholder(tf.float32, [None, x_features])
            self.y_placeholder = tf.placeholder(tf.float32, [None, y_features])

            # Setup of the requisite number of layers, kept in a list
            i = 0
            self.graph_layers = []
            self.graph_layers.append(tf.layers.dense(self.x_placeholder, layer_sizes[i], activation=tf.nn.relu))
            i += 1
            while i < hidden_layers:
                self.graph_layers.append(tf.layers.dense(self.graph_layers[i - 1], layer_sizes[i],
                                                         activation=tf.nn.relu))
                i += 1

            # Setup of the outputs, with support for two constants todo: need to be able to change the activation fn
            self.mixture_weights = tf.layers.dense(self.graph_layers[-1], mixture_components, activation=tf.nn.softmax)
            self.constant_one = tf.layers.dense(self.graph_layers[-1], mixture_components, activation=None)
            self.constant_two = tf.layers.dense(self.graph_layers[-1], mixture_components, activation=tf.exp)

            # Initialise the loss function and training scheme todo: only supports the default
            self.loss_function = loss_funcs.normal_distribution_l0_reg(self.y_placeholder, self.mixture_weights,
                                                                       self.constant_one, self.constant_two)
            self.train_function = tf.train.AdamOptimizer().minimize(self.loss_function)

            # Initialise a tensorflow session object using our lovely graph we just made, and initialise the variables
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

        # Create a blank loss array we can append more stuff to later for recording the loss function evolution
        self.loss = np.array([])

        # Initialise a blank feed dictionary dictionary (that's a mouthful... I'm so sorry)
        self.dict_of_feed_dicts = {}

    def __del__(self):
        """Closes the tensorflow session."""
        self.session.close()

    # todo: everything below here

    def add_new_data_source(self, x_data, y_data, name: str):
        """Modifies the class-unique feed dictionary to give to the network. Used to change the data being worked on.
        Needs to be given a name so that you can specify the data to be ran on later.

        Args:
            x_data (any): independent variables to feed to the network.
            y_data (any): dependent variables to feed to the network.
            name (str): name to give to this data source.
        """
        # Add the new x_data, y_data
        self.dict_of_feed_dicts[str(name)] = {self.x_placeholder: x_data, self.y_placeholder: y_data}

    def train(self, data_source: str, iterations: int=50):
        """Trains the tensorflow graph for the specified amount of time."""
        # optimise: is there a way to record loss in-place in the loss function, instead of calling .run() twice?
        # Let the user know about our grand plans




        pass

    def plot_loss_function_evolution(self, start: int=0):
        """Returns a plot of the change of the loss function over time."""
        pass


# Unit tests: implements the class on the toy_mdn_emily example data, using data from the following blog post:
# http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
if __name__ == '__main__':
    # todo: this unit test
    pass
