"""A Python file storing different potential loss functions for use in the code. Add new ones as you please!"""

import tensorflow as tf
import numpy as np

# Define 1 / sqrt(2pi) for future use
one_div_sqrt_two_pi = 1 / np.sqrt(2 * np.pi)


# Different mixtures written in tensorflow
def tf_mixture_normal(a_point, my_means, my_std_deviations):
    """Normal distribution implemented in tensorflow notation."""
    result = tf.subtract(a_point, my_means)
    result = tf.multiply(result, tf.reciprocal(my_std_deviations))
    result = -tf.square(result)/2
    return tf.multiply(tf.exp(result), tf.reciprocal(my_std_deviations)) * one_div_sqrt_two_pi


def tf_mixture_beta(a_point, my_a, my_b):
    # todo: write a beta distribution
    pass


# Different loss functions, using the above mixtures
def normal_distribution_l0_reg(a_point, my_weights, my_means, my_std_deviations):
    """Lossfunc defined in tensorflow notation."""  # todo: this docstring is shit
    # Calculate normal distribution mixture and normalise
    result = tf_mixture_normal(a_point, my_means, my_std_deviations)
    result = tf.multiply(result, my_weights)

    # Sum the result and take the mean negative log
    # todo: mean log is sensitive to outliers
    # todo: add some kind of L2 regularisation, penalising poor weight choices
    result = tf.reduce_sum(result, 1, keepdims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)

