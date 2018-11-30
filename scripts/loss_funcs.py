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


class NormalPDFLoss:

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

    @staticmethod
    def tensor_evaluate(true_values, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            true_values: the y/dependent variable data to evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        with tf.variable_scope('loss_func_evaluation'):
            # Initialise all of our lovely distribution friends
            distributions = tf.distributions.Normal(coefficients['means'], coefficients['std_deviations'])

            # Tile the true values so that they have the same shape as the distributions and can be evaluated
            tiled_true_values = tf.tile(true_values, [1, tf.shape(coefficients['means'])[1]])

            # Evaluate the PDF of all the distributions, then do some summing to find the PDF of the overall mixtures
            weighted_pdfs = tf.multiply(distributions.prob(tiled_true_values), coefficients['weights'])
            weighted_pdfs = tf.reduce_sum(weighted_pdfs, axis=1, keepdims=False)

            # Take the mean negative log and return
            log_pdfs = -tf.log(weighted_pdfs)
            return tf.reduce_mean(log_pdfs)

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
        i = 0
        for a_weight, a_mean, a_std_d in zip(coefficients['weights'], coefficients['means'],
                                             coefficients['std_deviations']):
            all_the_pdfs[i, :] = scipy_normal.pdf(x_data, loc=a_mean, scale=a_std_d) * a_weight
            i += 1

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

    @staticmethod
    def draw_random_variables(n_samples: int, coefficients: dict):
        """Draws random variables from pdf_multiple_points for numerical integration uses. May only be used on a single
        mixture at a time (so make sure elements of 'coefficients' are 1D.) Otherwise, it will fail badly, and quietly
        too (no error checking is added to make sure it stays fast.)

        Args:
            n_samples (int): how many samples to draw.
            coefficients (dict): validation dictionary specifying the details of the mixture pdf.

        Returns:
            a np.array containing samples drawn from the mixture.
        """
        # Define a dictionary of samples and work out how many we'll want to draw from each mixture
        random_variables = np.empty(n_samples)
        number_weights = np.around(coefficients['weights'] * n_samples).astype(int)

        # Add 0 to the start of number_weights so that when writing random variables we start at the beginning of
        # the array
        number_weights = np.append([0], number_weights)

        # If number_weights sums to more than the number of points specified (due to a rounding error) then subtract the
        # rounding error (not normally more than just 1) from the largest weights.
        difference = n_samples - np.sum(number_weights)

        if difference is not 0:
            number_weights[np.argmax(number_weights)] += difference

        # Make a cumulative sum so that we've got more by way of
        number_indices = np.cumsum(number_weights)

        # Draw some random variables for fun
        i = 1
        for a_mean, a_std_d in zip(coefficients['means'], coefficients['std_deviations']):
            random_variables[number_indices[i-1]:number_indices[i]] = scipy_normal.rvs(loc=a_mean, scale=a_std_d,
                                                                                       size=number_weights[i])
            i += 1

        return random_variables


class BetaPDFLoss:

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
        self.kernel_initializers = [tf.initializers.random_uniform, tf.initializers.random_uniform,
                                    tf.initializers.random_uniform]
        self.bias_initializers = [tf.initializers.zeros, tf.initializers.zeros,
                                  tf.initializers.zeros]

    @staticmethod
    def tensor_evaluate(true_values, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            true_values: the y/dependent variable data to tensor_evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        with tf.variable_scope('loss_func_evaluation'):
            # Initialise all of our lovely distribution friends
            distributions = tf.distributions.Beta(coefficients['alpha'], coefficients['beta'])

            # Tile the true values so that they have the same shape as the distributions and can be evaluated
            tiled_true_values = tf.tile(true_values, [1, tf.shape(coefficients['alpha'])[1]])

            # Evaluate the PDF of all the distributions, then do some summing to find the PDF of the overall mixtures
            weighted_pdfs = tf.multiply(distributions.prob(tiled_true_values), coefficients['weights'])
            weighted_pdfs = tf.reduce_sum(weighted_pdfs, axis=1, keepdims=False)

            # Take the mean negative log and return
            log_pdfs = -tf.log(weighted_pdfs)
            return tf.reduce_mean(log_pdfs)

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
        i = 0
        for a_weight, a_alpha, a_beta in zip(coefficients['weights'], coefficients['alpha'],
                                             coefficients['beta']):
            all_the_pdfs[i, :] = scipy_beta.pdf(x_data, a_alpha, a_beta) * a_weight
            i += 1

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

    @staticmethod
    def draw_random_variables(n_samples: int, coefficients: dict):
        """Draws random variables from pdf_multiple_points for numerical integration uses. May only be used on a single
        mixture at a time (so make sure elements of 'coefficients' are 1D.) Otherwise, it will fail badly, and quietly
        too (no error checking is added to make sure it stays fast.)

        Args:
            n_samples (int): how many samples to draw.
            coefficients (dict): validation dictionary specifying the details of the mixture pdf.

        Returns:
            a np.array containing samples drawn from the mixture.
        """
        # Define a dictionary of samples and work out how many we'll want to draw from each mixture
        random_variables = np.empty(n_samples)
        number_weights = np.around(coefficients['weights'] * n_samples).astype(int)

        # Add 0 to the start of number_weights so that when writing random variables we start at the beginning of
        # the array
        number_weights = np.append([0], number_weights)

        # If number_weights sums to more than the number of points specified (due to a rounding error) then subtract the
        # rounding error (not normally more than just 1) from the largest weights.
        difference = n_samples - np.sum(number_weights)

        if difference is not 0:
            number_weights[np.argmax(number_weights)] += difference

        # Make a cumulative sum so that we've got more by way of
        number_indices = np.cumsum(number_weights)

        # Draw some random variables for fun
        i = 1
        for a_alpha, a_beta in zip(coefficients['alpha'], coefficients['beta']):
            random_variables[number_indices[i - 1]:number_indices[i]] = scipy_beta.rvs(a_alpha, a_beta,
                                                                                       size=number_weights[i])
            i += 1

        return random_variables


class BetaCDFLoss:

    def __init__(self):
        """Creates a beta distribution implemented in tensorflow notation. CAUTION: it's approximate!  # todo: this fnstring
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """

        raise NotImplementedError('Beta CDF hasn\t been implemented, silly!')
        # Names of and activation functions to apply to each output layer.
        self.coefficient_names = ['weights', 'alpha', 'beta']
        self.activation_functions = [tf.nn.softmax, tf.exp, tf.exp]

        # Variable initializers for the weights (kernel) and biases of each output layer. Try to set them sensibly.
        self.kernel_initializers = [tf.initializers.random_uniform, tf.initializers.random_uniform,
                                    tf.initializers.random_uniform]
        self.bias_initializers = [tf.initializers.zeros, tf.initializers.zeros,
                                  tf.initializers.zeros]

    def tensor_evaluate(self, true_values, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            true_values: the y/dependent variable data to tensor_evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        with tf.variable_scope('loss_func_evaluation'):
            """
            # Initialise all of our lovely beta distribution friends
            distributions = tf.distributions.Beta(coefficients['alpha'], coefficients['beta'])

            # Tile the true values so that they have the same shape as the beta distributions and can be evaluated
            tiled_true_values = tf.tile(true_values, [1, tf.shape(coefficients['alpha'])[1]])

            # Evaluate the CDF of all the distributions, then do some summing to find the CDF of the overall mixtures
            weighted_cdfs = tf.multiply(distributions.cdf(tiled_true_values), coefficients['weights'])
            weighted_cdfs = tf.reduce_sum(weighted_cdfs, axis=1, keepdims=False)

            # Sort and cumulatively sum the result, weighting it with the total sum of the cdfs so that the maximum val
            # in summed_cdfs is 1.0
            sorted_cdfs = tf.contrib.framework.sort(weighted_cdfs, axis=0)
            summed_cdfs = tf.math.cumsum(sorted_cdfs, axis=0)
            summed_cdfs = tf.divide(summed_cdfs, summed_cdfs[-1])

            # Minimise the residual between the cumulative sum of cdf evaluations and a linear series of numbers, that
            # we would expect the cumulative sum to take if it was correct
            expected_cdfs = tf.linspace(0.0, 1.0, num=tf.shape(summed_cdfs)[0])  # optimise: this is declared every time
            cdf_residuals = tf.square(tf.subtract(summed_cdfs, expected_cdfs))
            """

            # Next, find the modes of the distributions and see if they're far away from the true values
            top = tf.subtract(coefficients['alpha'], 1.)
            bottom = tf.subtract(tf.add(coefficients['alpha'], coefficients['beta']), 2.)
            inferred_modes = tf.divide(top, bottom)
            deltas = tf.subtract(inferred_modes, true_values)
            mode_residuals = tf.square(tf.divide(deltas, tf.add(1., true_values)))

            return tf.log(tf.reduce_mean(mode_residuals))

            # Sum and return the log-result
            #return tf.log(tf.multiply(tf.reduce_sum(cdf_residuals), tf.reduce_mean(mode_residuals)))

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
        i = 0
        for a_weight, a_alpha, a_beta in zip(coefficients['weights'], coefficients['alpha'],
                                             coefficients['beta']):
            all_the_pdfs[i, :] = scipy_beta.pdf(x_data, a_alpha, a_beta) * a_weight
            i += 1

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

    @staticmethod
    def draw_random_variables(n_samples: int, coefficients: dict):
        """Draws random variables from pdf_multiple_points for numerical integration uses. May only be used on a single
        mixture at a time (so make sure elements of 'coefficients' are 1D.) Otherwise, it will fail badly, and quietly
        too (no error checking is added to make sure it stays fast.)

        Args:
            n_samples (int): how many samples to draw.
            coefficients (dict): validation dictionary specifying the details of the mixture pdf.

        Returns:
            a np.array containing samples drawn from the mixture.
        """
        # Define a dictionary of samples and work out how many we'll want to draw from each mixture
        random_variables = np.empty(n_samples)
        number_weights = np.around(coefficients['weights'] * n_samples).astype(int)

        # Add 0 to the start of number_weights so that when writing random variables we start at the beginning of
        # the array
        number_weights = np.append([0], number_weights)

        # If number_weights sums to more than the number of points specified (due to a rounding error) then subtract the
        # rounding error (not normally more than just 1) from the largest weights.
        difference = n_samples - np.sum(number_weights)

        if difference is not 0:
            number_weights[np.argmax(number_weights)] += difference

        # Make a cumulative sum so that we've got more by way of
        number_indices = np.cumsum(number_weights)

        # Draw some random variables for fun
        i = 1
        for a_alpha, a_beta in zip(coefficients['alpha'], coefficients['beta']):
            random_variables[number_indices[i - 1]:number_indices[i]] = scipy_beta.rvs(a_alpha, a_beta,
                                                                                       size=number_weights[i])
            i += 1

        return random_variables


class NormalCDFLoss:

    def __init__(self, cdf_strength: float=0.1):
        """Creates a normal distribution implemented in tensorflow notation.
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.

        Args:
            - error_strength (float): strength of the error test in this lossfunc.
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

        # Store variables for later
        self.error_strength = cdf_strength

    def tensor_evaluate(self, true_values, coefficients):
        """Lossfunc defined in tensorflow notation.

        Args:
            true_values: the y/dependent variable data to evaluate against.
            coefficients: a dictionary with a 'weights', 'means' and 'std_deviations' argument.

        Returns:
            a float giving the loss
        """
        with tf.variable_scope('loss_func_evaluation'):
            # DISTRIBUTION CONFIDENCE AND SYSTEMATIC BIAS EVALUATION
            # Initialise all of our lovely beta distribution friends
            distributions = tf.distributions.Normal(coefficients['means'], coefficients['std_deviations'])

            # Tile the true values so that they have the same shape as the beta distributions and can be evaluated
            tiled_true_values = tf.tile(true_values, [1, tf.shape(coefficients['means'])[1]])

            # Evaluate the CDF of all the distributions, then do some summing to find the CDF of the overall mixtures
            weighted_cdfs = tf.multiply(distributions.cdf(tiled_true_values), coefficients['weights'])
            weighted_cdfs = tf.reduce_sum(weighted_cdfs, axis=1, keepdims=False)

            # Sort and cumulatively sum the result, weighting it with the total sum of the cdfs so that the maximum val
            # in summed_cdfs is 1.0
            sorted_cdfs = tf.contrib.framework.sort(weighted_cdfs, axis=0)
            summed_cdfs = tf.math.cumsum(weighted_cdfs, axis=0)
            summed_cdfs = tf.divide(summed_cdfs, summed_cdfs[-1])

            # Calculate the mean squared residual between summed cdfs and sorted cdfs
            cdf_residual = tf.multiply(tf.reduce_mean(tf.log(tf.cosh(tf.subtract(summed_cdfs, sorted_cdfs)))),
                                       self.error_strength)

            # DISTRIBUTION ACCURACY EVALUATION
            # map_residual = tf.log(tf.reduce_mean(tf.square(tf.subtract(coefficients['means'][:, 0], true_values[:, 0]))))

            # Calculate normal distribution mixture and normalise
            weighted_pdfs = tf.multiply(distributions.prob(tiled_true_values), coefficients['weights'])

            # Sum the result and take the negative mean
            summed_pdfs = tf.reduce_sum(weighted_pdfs, 1, keepdims=False)
            mean_log_pdf = tf.reduce_mean(tf.log(summed_pdfs))
            pdf_residual = tf.multiply(mean_log_pdf, -1)

            #cdf_residual = tf.constant(1.0)

            return tf.add(cdf_residual, pdf_residual)


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
        i = 0
        for a_weight, a_mean, a_std_d in zip(coefficients['weights'], coefficients['means'],
                                             coefficients['std_deviations']):
            all_the_pdfs[i, :] = scipy_normal.pdf(x_data, loc=a_mean, scale=a_std_d) * a_weight
            i += 1

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

    @staticmethod
    def draw_random_variables(n_samples: int, coefficients: dict):
        """Draws random variables from pdf_multiple_points for numerical integration uses. May only be used on a single
        mixture at a time (so make sure elements of 'coefficients' are 1D.) Otherwise, it will fail badly, and quietly
        too (no error checking is added to make sure it stays fast.)

        Args:
            n_samples (int): how many samples to draw.
            coefficients (dict): validation dictionary specifying the details of the mixture pdf.

        Returns:
            a np.array containing samples drawn from the mixture.
        """
        # Todo: I fail really badly when passed a python list (it makes number_weights into a longer list, it's owwie)

        # Define a dictionary of samples and work out how many we'll want to draw from each mixture
        random_variables = np.empty(n_samples)
        number_weights = np.around(coefficients['weights'] * n_samples).astype(int)

        # Add 0 to the start of number_weights so that when writing random variables we start at the beginning of
        # the array
        number_weights = np.append([0], number_weights)

        # If number_weights sums to more than the number of points specified (due to a rounding error) then subtract the
        # rounding error (not normally more than just 1) from the largest weights.
        difference = n_samples - np.sum(number_weights)

        if difference is not 0:
            number_weights[np.argmax(number_weights)] += difference

        # Make a cumulative sum so that we've got more by way of
        number_indices = np.cumsum(number_weights)

        # Draw some random variables for fun
        i = 1
        for a_mean, a_std_d in zip(coefficients['means'], coefficients['std_deviations']):
            random_variables[number_indices[i-1]:number_indices[i]] = scipy_normal.rvs(loc=a_mean, scale=a_std_d,
                                                                                       size=number_weights[i])
            i += 1

        return random_variables