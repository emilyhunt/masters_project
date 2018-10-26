"""A Python file storing different potential loss functions for use in the code. Add new ones as you please!

Note that as currently implemented, loss functions should:
    - Be a class, with a list self.activation_functions of different activation functions that should be used with
      each different mixture co-efficient
    - Have an 'evaluate' argument that a MixtureDensityNetwork class can use, with:
        ~ an argument that is 'a point' aka the y/dependent variable data
        ~ the second argument should be a dictionary containing all of the required co-efficients

"""

import tensorflow as tf
import numpy as np


class NormalDistribution:

    def __init__(self):
        """Creates a normal distribution implemented in tensorflow notation.
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """
        self.activation_functions = [tf.nn.softmax, None, tf.exp]
        self.coefficient_names = ['weights', 'means', 'std_deviations']

        # A useful constant to keep around
        self.one_div_sqrt_two_pi = 1 / np.sqrt(2 * np.pi)

    def tensor_normal_distribution(self, a_point, my_means, my_std_deviations):
        """A normal distribution implemented in tensorflow notation."""
        result = tf.subtract(a_point, my_means)
        result = tf.multiply(result, tf.reciprocal(my_std_deviations))
        result = -tf.square(result) / 2
        return tf.multiply(tf.exp(result), tf.reciprocal(my_std_deviations)) * self.one_div_sqrt_two_pi

    def evaluate(self, a_point, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            a_point: the y/dependent variable data to evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the mean
        """
        # Calculate normal distribution mixture and normalise
        result = self.tensor_normal_distribution(a_point, coefficients['means'], coefficients['std_deviations'])
        result = tf.multiply(result, coefficients['weights'])

        # Sum the result and take the mean negative log
        # todo: mean log is sensitive to outliers
        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)


class BetaDistribution:

    def __init__(self):
        """Creates a beta distribution implemented in tensorflow notation. CAUTION: it's approximate!
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """
        self.activation_functions = [tf.nn.softmax, tf.exp, tf.exp]
        self.coefficient_names = ['weights', 'alpha', 'beta']

        # A useful constant to keep around
        self.one_div_sqrt_two_pi = 1 / np.sqrt(2 * np.pi)

    @staticmethod
    def tensor_log_gamma(x):
        """A fast approximate log gamma function from Paul Mineiro. See:
        http://www.machinedlearnings.com/2011/06/faster-lda.html
        """
        log_term = tf.log(x * (1.0 + x) * (2.0 + x))
        x_plus_3 = 3.0 + x
        return -2.081061466 - x + 0.0833333 / x_plus_3 - log_term + (2.5 + x) * tf.log(x_plus_3)

    def tensor_beta_distribution(self, a_point, alpha, beta):
        """An APPROXIMATE (& fast) beta distribution implemented in tensorflow notation."""
        exp1 = tf.subtract(alpha, 1.0)
        exp2 = tf.subtract(beta, 1.0)
        d1 = tf.multiply(exp1, tf.log(a_point))
        d2 = tf.multiply(exp2, tf.log(tf.subtract(1.0, a_point)))
        f1 = tf.add(d1, d2)
        f2 = self.tensor_log_gamma(alpha)
        f3 = self.tensor_log_gamma(beta)
        f4 = self.tensor_log_gamma(alpha + beta)
        return tf.exp(tf.add((tf.subtract(f4, tf.add(f2, f3))), f1))

    def evaluate(self, a_point, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            a_point: the y/dependent variable data to evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the mean
        """
        # Calculate normal distribution mixture and normalise
        result = self.tensor_beta_distribution(a_point, coefficients['alpha'], coefficients['beta'])
        result = tf.multiply(result, coefficients['weights'])

        # Sum the result and take the mean negative log
        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)
