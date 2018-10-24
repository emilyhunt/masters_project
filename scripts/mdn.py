"""Placeholder for a potential future class, depending on how I go about things."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scripts import loss_funcs


class MixtureDensityNetwork:

    def __init__(self, loss_function, x_features=1, y_features=1, hidden_layers=2, layer_sizes=15,
                 mixture_components=5):
        """Initialises an MDN in tensorflow given the specified (or default) parameters."""
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

    def add_new_data_source(self, x_data, y_data, name):
        """Modifies the class-unique feed dictionary to give to the network. Used to change the data being worked on.
        Needs to be given a name so that you can specify the data to be ran on later.
        """
        pass

    def run(self, data_source, iterations=50):
        """Runs a tensorflow session for the specified amount of time."""
        # optimise: is there a way to record loss in-place in the loss function, instead of calling .run() twice?
        pass

    def plot_loss_function_evolution(start=0):
        """Returns a plot of the change of the loss function over time."""
        pass