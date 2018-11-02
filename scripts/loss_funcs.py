"""A Python file storing different potential loss functions for use in the code. Add new ones as you please!

Note that as currently implemented, loss functions should:
    - Be a class, with a list self.activation_functions of different activation functions that should be used with
      each different mixture co-efficient
    - Have an 'tensor_evaluate' argument that a MixtureDensityNetwork class can use, with:
        ~ an argument that is 'a point' aka the y/dependent variable data
        ~ the second argument should be a dictionary containing all of the required co-efficients

"""

import tensorflow as tf
import numpy as np
from scipy.stats import norm as scipy_normal
from scipy.stats import beta as scipy_beta
from typing import Union


class NormalDistribution:

    def __init__(self):
        """Creates a normal distribution implemented in tensorflow notation.
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """
        # Names of and activation functions to apply to each output layer.
        self.coefficient_names = ['weights', 'means', 'std_deviations']
        self.activation_functions = [tf.nn.softmax, None, tf.exp]

        # Variable initializers for the weights (kernel) and biases of each output layer. Try to set them sensibly.
        self.kernel_initializers = [tf.initializers.random_normal, tf.initializers.random_normal,
                                    tf.initializers.ones]
        self.bias_initializers = [tf.initializers.ones, tf.initializers.ones, tf.initializers.ones]

        # A useful constant to keep around
        self.one_div_sqrt_two_pi = np.float64(1 / np.sqrt(2 * np.pi))

    def tensor_normal_distribution(self, a_point, my_means, my_std_deviations):
        """A normal distribution implemented in tensorflow notation."""
        result = tf.subtract(a_point, my_means)
        result = tf.multiply(result, tf.reciprocal(my_std_deviations))
        result = -tf.square(result) / np.float64(2)
        return tf.multiply(tf.exp(result), tf.reciprocal(my_std_deviations)) * self.one_div_sqrt_two_pi

    def tensor_evaluate(self, a_point, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            a_point: the y/dependent variable data to evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        # Calculate normal distribution mixture and normalise
        result = self.tensor_normal_distribution(a_point, coefficients['means'], coefficients['std_deviations'])
        result = tf.multiply(result, coefficients['weights'])

        # Sum the result and take the mean negative log
        # todo: mean log is sensitive to outliers
        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)

    @staticmethod
    def pdf_multiple_points(x_data, coefficients: dict, sum_mixtures=True):
        """Lossfunc defined in simple numpy arrays. Will instead return a pdf for the given object. Fastest at
        evaluating lots of x data points: for instance, for initial minimum finding or for plotting a pdf.

                Args:
                    x_data: the y/dependent variable data to evaluate against. Can be an array or a float.
                    coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                                  of those values for the given object.
                    sum_mixtures: defines whether or not we should sum the mixture distribution at each point or return
                                  a big array of all the mixture components individually.

                Returns:
                    A 1D array of the pdf value(s) at whatever point(s) you've evaluated it at.
        """
        # Typecast x_data as a 1D numpy array and work out how many mixtures there are
        x_data = np.array([x_data]).flatten()
        n_mixtures = coefficients['weights'].size
        all_the_pdfs = np.empty((n_mixtures, x_data.size))

        # Cycle over each mixture and evaluate the pdf
        for i, a_weight, a_mean, a_std_d in zip(enumerate(coefficients['weights']), coefficients['means'],
                                                      coefficients['std_deviations']):
            all_the_pdfs[i, :] = scipy_normal.pdf(x_data, loc=a_mean, scale=a_std_d) * a_weight

        # Sum if requested
        if sum_mixtures:
            return np.sum(all_the_pdfs, axis=0)
        else:
            return all_the_pdfs

    @staticmethod
    def pdf_single_point(x_point: Union[float, int], coefficients: dict):
        """Lossfunc implemented in a much faster way for single point evaluations.
        Args:
            x_point: the y/dependent variable data to evaluate against. Can only be a float or int.
            coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                          of those values for the given object.

        Returns:
            A float of the pdf evaluated at a single point.
        """
        return np.sum(scipy_normal.pdf(x_point, loc=coefficients['means'], scale=coefficients['std_deviations'])
                      * coefficients['weights'])


class BetaDistribution:

    def __init__(self):
        """Creates a beta distribution implemented in tensorflow notation. CAUTION: it's approximate!
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """
        # Names of and activation functions to apply to each output layer.
        self.coefficient_names = ['weights', 'alpha', 'beta']
        self.activation_functions = [tf.nn.softmax, tf.exp, tf.exp]

        # Variable initializers for the weights (kernel) and biases of each output layer. Try to set them sensibly.
        self.kernel_initializers = [tf.initializers.random_normal, tf.initializers.random_uniform,
                                    tf.initializers.random_uniform]
        self.bias_initializers = [tf.initializers.ones, tf.initializers.zeros, tf.initializers.zeros]

    @staticmethod
    def tensor_log_gamma(x):
        """A fast approximate log gamma function from Paul Mineiro. See:
        http://www.machinedlearnings.com/2011/06/faster-lda.html
        """
        log_term = tf.log(x * (np.float64(1.0) + x) * (np.float64(2.0) + x))
        x_plus_3 = np.float64(3.0) + x
        return np.float64(-2.081061466) - x + np.float64(0.0833333) / x_plus_3 - log_term + (np.float64(2.5) + x) * tf.log(x_plus_3)

    def tensor_beta_distribution(self, a_point, alpha, beta):
        """An APPROXIMATE (& fast) beta distribution implemented in tensorflow notation."""
        exp1 = tf.subtract(alpha, np.float64(1.0))
        exp2 = tf.subtract(beta, np.float64(1.0))
        d1 = tf.multiply(exp1, tf.log(a_point))
        d2 = tf.multiply(exp2, tf.log(tf.subtract(np.float64(1.0), a_point)))
        f1 = tf.add(d1, d2)
        f2 = self.tensor_log_gamma(alpha)
        f3 = self.tensor_log_gamma(beta)
        f4 = self.tensor_log_gamma(alpha + beta)
        return tf.exp(tf.add((tf.subtract(f4, tf.add(f2, f3))), f1))

    def tensor_evaluate(self, a_point, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            a_point: the y/dependent variable data to tensor_evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        # Calculate normal distribution mixture and normalise
        result = self.tensor_beta_distribution(a_point, coefficients['alpha'], coefficients['beta'])
        result = tf.multiply(result, coefficients['weights'])

        # Sum the result and take the mean negative log
        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)

    @staticmethod
    def pdf_multiple_points(x_data, coefficients: dict, sum_mixtures=True):
        """Lossfunc defined in simple numpy arrays. Will instead return a pdf for the given object. Fastest at
        evaluating lots of x data points: for instance, for initial minimum finding or for plotting a pdf.

                Args:
                    x_data: the y/dependent variable data to evaluate against. Can be an array or a float.
                    coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                                  of those values for the given object.
                    sum_mixtures: defines whether or not we should sum the mixture distribution at each point or return
                                  a big array of all the mixture components individually.

                Returns:
                    A 1D array of the pdf value(s) at whatever point(s) you've evaluated it at.
        """
        # Typecast x_data as a 1D numpy array and work out how many mixtures there are
        x_data = np.array([x_data]).flatten()
        n_mixtures = coefficients['weights'].size
        all_the_pdfs = np.empty((n_mixtures, x_data.size))

        # Cycle over each mixture and evaluate the pdf
        for i, a_weight, a_alpha, a_beta in enumerate(zip(coefficients['weights'], coefficients['alpha'],
                                                      coefficients['beta'])):
            all_the_pdfs[i, :] = scipy_beta.pdf(x_data, a_alpha, a_beta) * a_weight

        # Sum if requested
        if sum_mixtures:
            return np.sum(all_the_pdfs, axis=0)
        else:
            return all_the_pdfs

    @staticmethod
    def pdf_single_point(x_point: Union[float, int], coefficients: dict):
        """Lossfunc implemented in a much faster way for single point evaluations.
        Args:
            x_point: the y/dependent variable data to evaluate against. Can only be a float or int.
            coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                          of those values for the given object.

        Returns:
            A float of the pdf evaluated at a single point.
        """
        return np.sum(scipy_beta.pdf(x_point, coefficients['alpha'], coefficients['beta']) * coefficients['weights'])
