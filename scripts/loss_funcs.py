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
from typing import Union, Optional


class NormalPDFLoss:

    def __init__(self, perturbation_coefficient_0: float=0.3, perturbation_coefficient_1: float=0.0,
                 perturbation_coefficient_2: float=0.0, perturbation_coefficient_3: float=0.0):
        """Creates a normal distribution implemented in tensorflow notation.
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.
        """
        # Names of and activation functions to apply to each output layer.
        self.coefficient_names = ['weights', 'means', 'std_deviations']
        self.activation_functions = [tf.nn.softmax, binary_activation, tf.exp]

        # Variable initializers for the weights (kernel) and biases of each output layer. Try to set them sensibly.
        self.kernel_initializers = [None, None, None]
        self.bias_initializers = [tf.initializers.zeros, tf.initializers.ones, tf.initializers.ones]

        # A useful constant to keep around
        self.perturbation_coefficient_0 = perturbation_coefficient_0
        self.perturbation_coefficient_1 = perturbation_coefficient_1
        self.perturbation_coefficient_2 = perturbation_coefficient_2
        self.perturbation_coefficient_3 = perturbation_coefficient_3

    def tensor_evaluate(self, true_values, coefficients):
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

            # Perturb all of the true values with our perturbation law
            random_values = tf.random.normal(tf.shape(true_values), mean=0.0, stddev=1.0)
            true_values_squared = tf.square(true_values)
            true_values_cubed = tf.multiply(true_values, true_values_squared)

            perturbation_0 = tf.multiply(random_values, self.perturbation_coefficient_0)
            perturbation_1 = tf.multiply(tf.multiply(random_values, self.perturbation_coefficient_1),
                                         true_values)
            perturbation_2 = tf.multiply(tf.multiply(random_values, self.perturbation_coefficient_2),
                                         true_values_squared)
            perturbation_3 = tf.multiply(tf.multiply(random_values, self.perturbation_coefficient_3),
                                         true_values_cubed)

            true_values = tf.add(tf.add(tf.add(tf.add(perturbation_0, perturbation_1),
                                               perturbation_2), perturbation_3), true_values)

            # Tile the true values so that they have the same shape as the distributions and can be evaluated
            tiled_true_values = tf.tile(true_values, [1, tf.shape(coefficients['means'])[1]])

            # Evaluate the PDF of all the distributions, then do some summing to find the PDF of the overall mixtures
            weighted_pdfs = tf.multiply(distributions.prob(tiled_true_values), coefficients['weights'])
            weighted_pdfs = tf.reduce_sum(weighted_pdfs, axis=1, keepdims=False)

            # Calculate some stats
            max_pdf = tf.reduce_max(weighted_pdfs)
            min_pdf = tf.reduce_min(weighted_pdfs)
            mean_pdf = tf.reduce_mean(weighted_pdfs)

            max_mean = tf.reduce_max(coefficients['means'])
            min_mean = tf.reduce_min(coefficients['means'])
            mean_mean = tf.reduce_mean(coefficients['means'])

            max_std = tf.reduce_max(coefficients['std_deviations'])
            min_std = tf.reduce_min(coefficients['std_deviations'])
            mean_std = tf.reduce_mean(coefficients['std_deviations'])

            max_weight = tf.reduce_max(coefficients['weights'])
            min_weight = tf.reduce_min(coefficients['weights'])
            mean_weight = tf.reduce_mean(coefficients['weights'])

            # Take the mean negative log and return
            log_pdfs = tf.reduce_mean(tf.multiply(-1., tf.log(weighted_pdfs)))
            return {'total_residual': log_pdfs,
                    'pdf_max': max_pdf, 'pdf_min': min_pdf, 'pdf_mean': mean_pdf,
                    'mean_max': max_mean, 'mean_min': min_mean, 'mean_mean': mean_mean,
                    'std_max': max_std, 'std_min': min_std, 'std_mean': mean_std,
                    'weight_max': max_weight, 'weight_min': min_weight, 'weight_mean': mean_weight}

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

        # Sum each mixture if requested, giving the cdf of the overall mixture density for that object
        if sum_mixtures:
            return np.sum(all_the_pdfs, axis=0)
        else:
            return all_the_pdfs

    @staticmethod
    def cdf_multiple_points(x_data, coefficients: dict, sum_mixtures=True):
        """Lossfunc defined in simple numpy arrays. Will instead return a cdf for the given object. Fastest at
        evaluating lots of x data points: for instance, for initial minimum finding or for plotting a cdf.

            Args:
                x_data: the y/dependent variable data to evaluate against. Can be an array or a float.
                coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                              of those values for the given object.
                sum_mixtures: defines whether or not we should sum the mixture distribution at each point or return
                              a big array of all the mixture components individually.

            Returns:
                A 1D array of the cdf value(s) at whatever point(s) you've evaluated it at.
        """
        # Typecast x_data as a 1D numpy array and work out how many mixtures there are
        x_data = np.array([x_data]).flatten()
        n_mixtures = coefficients['weights'].size
        all_the_cdfs = np.empty((n_mixtures, x_data.size))

        # Cycle over each mixture and evaluate the pdf
        i = 0
        for a_weight, a_mean, a_std_d in zip(coefficients['weights'], coefficients['means'],
                                             coefficients['std_deviations']):
            all_the_cdfs[i, :] = scipy_normal.cdf(x_data, loc=a_mean, scale=a_std_d) * a_weight
            i += 1

        # Sum each mixture if requested, giving the cdf of the overall mixture density for that object
        if sum_mixtures:
            return np.sum(all_the_cdfs, axis=0)
        else:
            return all_the_cdfs

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
    def cdf_single_point(x_point: Union[float, int], coefficients: dict):
        """CDF of lossfunc implemented in a much faster way for single point evaluations.
        Args:
            x_point: the y/dependent variable data to evaluate against. Can only be a float or int.
            coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                          of those values for the given object.

        Returns:
            A float of the cdf evaluated at a single point.
        """
        return np.sum(scipy_normal.cdf(x_point, loc=coefficients['means'], scale=coefficients['std_deviations'])
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


def binary_activation(x):

    cond = tf.less(x, 0.)
    out = tf.where(cond, tf.zeros(tf.shape(x)), x)

    cond = tf.greater(out, 7.)
    out = tf.where(cond, tf.multiply(7., tf.ones(tf.shape(x))), out)

    return out

def binary_activation_2(x):

    #x = tf.exp(x)
    cond = tf.less(x, 0.05)
    out = tf.where(cond, tf.multiply(0.05, tf.ones(tf.shape(x))), x)

    cond = tf.greater(x, 10.)
    out = tf.where(cond, tf.multiply(10., tf.ones(tf.shape(x))), x)

    return out


class NormalCDFLoss:

    def __init__(self, cdf_strength: float=1.0, std_deviation_strength: float=1.0, normalisation_strength: float=1.0,
                 grid_size: Optional[int]=None, redshift_range: tuple=(0, 7), mixtures: int=1):
        """Creates a normal distribution implemented in tensorflow notation.
        You may wish to access:
            - self.activation_functions: a list of activation functions that should be used with the mixture arguments,
                                         applied to layers.
            - self.coefficient_names: a list of the names of each mixture coefficient.

        Args:
            - error_strength (float): strength of the error test in this lossfunc.  # todo: update me
        """
        # Names of and activation functions to apply to each output layer.
        self.coefficient_names = ['weights', 'means', 'std_deviations']
        self.activation_functions = [tf.nn.softmax, binary_activation, tf.exp]

        # Variable initializers for the weights (kernel) and biases of each output layer. Try to set them sensibly.
        self.kernel_initializers = [None, None, None]
        self.bias_initializers = [tf.initializers.zeros, tf.initializers.ones, tf.initializers.zeros]

        # A useful constant to keep around
        self.one_div_sqrt_two_pi = np.float64(1 / np.sqrt(2 * np.pi))

        # Store variables for later
        self.cdf_strength = cdf_strength
        self.std_deviation_strength = std_deviation_strength
        self.normalisation_strength = normalisation_strength
        self.redshift_range = redshift_range

        if grid_size is not None:
            self.grid = np.linspace(redshift_range[0], redshift_range[1], num=grid_size, dtype=np.float32)
            self.grid_size = grid_size
        else:
            self.grid = None
            self.grid_size = None

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

            # Evaluate the CDFs at the limits of the redshift range so we can normalise them
            cdf_constant = tf.multiply(distributions.cdf(self.redshift_range[0]), coefficients['weights'])
            cdf_constant = tf.reduce_sum(cdf_constant, axis=1, keepdims=False)
            cdf_multiplier = tf.multiply(distributions.cdf(self.redshift_range[1]), coefficients['weights'])
            cdf_multiplier = tf.reduce_sum(cdf_multiplier, axis=1, keepdims=False)

            # Normalise the existing CDFs
            weighted_cdfs = tf.subtract(tf.divide(weighted_cdfs, cdf_multiplier), cdf_constant)

            # Sort and cumulatively sum the result, weighting it with the total sum of the cdfs so that the maximum val
            # in summed_cdfs is 1.0
            sorted_cdfs = tf.contrib.framework.sort(weighted_cdfs, axis=0)

            expected_cdfs = tf.linspace(0.0, 1.0, num=tf.shape(sorted_cdfs)[0])

            #summed_cdfs = tf.math.cumsum(weighted_cdfs, axis=0)
            #summed_cdfs = tf.divide(summed_cdfs, summed_cdfs[-1])

            # Calculate the mean squared residual between summed cdfs and sorted cdfs
            cdf_residual = tf.multiply(tf.reduce_max(tf.log(tf.cosh(tf.subtract(expected_cdfs, sorted_cdfs)))),
                                       self.cdf_strength)

            # DISTRIBUTION ACCURACY EVALUATION
            # map_residual = tf.log(tf.reduce_mean(tf.square(tf.subtract(coefficients['means'][:, 0], true_values[:, 0]))))

            # Calculate normal distribution mixture and normalise against weights
            weighted_pdfs = tf.multiply(distributions.prob(tiled_true_values), coefficients['weights'])
            summed_pdfs = tf.reduce_sum(weighted_pdfs, 1, keepdims=False)

            # Calculate the maximum either with a grid or with an educated guess
            if self.grid is not None:
                # Tile up a grid
                the_shape = tf.shape(coefficients['means'])
                tiled_grid = tf.tile(tf.reshape(self.grid, [-1, 1, 1]), [1, the_shape[0], the_shape[1]])

                # Use tf.map_fun to evaluate the distributions across the grid
                distribution_function = lambda points: tf.multiply(distributions.prob(points), coefficients['weights'])
                tiled_distributions = tf.map_fn(distribution_function, tiled_grid)

                # Sum all of the mixtures and find the max of each combined pdf
                pdfs = tf.reduce_sum(tiled_distributions, axis=2)
                summed_pdf_maxes = tf.reduce_max(pdfs, axis=0)

            else:
                # Work out rough pdf maximums by evaluating at the mean of the most heavily weighted distributions
                # This is faster than gridding, and hopefully loses ~no accuracy since a grid has a finite accuracy anyway
                indices = tf.transpose([tf.range(0, tf.shape(coefficients['means'])[0]),
                                        tf.argmax(coefficients['weights'], axis=1, output_type=tf.int32)])
                biggest_pdfs = tf.expand_dims(tf.gather_nd(coefficients['means'], indices), axis=1)

                # Tile up the estimates of the pdf maxima and eval them against the distributions
                tiled_biggest_pdfs = tf.tile(biggest_pdfs, [1, tf.shape(coefficients['means'])[1]])
                approximate_pdf_max = tf.multiply(distributions.prob(tiled_biggest_pdfs), coefficients['weights'])
                summed_pdf_maxes = tf.reduce_sum(approximate_pdf_max, axis=1, keepdims=False)

            # Divide the pdfs by their maximum to make the pdf residuals standard-deviation invariant
            summed_pdf_maxes = tf.multiply(summed_pdf_maxes, self.normalisation_strength)
            summed_pdfs = tf.divide(summed_pdfs, tf.add(summed_pdf_maxes, tf.subtract(1., self.normalisation_strength)))

            # Sum the result and take the negative mean log
            mean_log_pdf = tf.reduce_mean(tf.log(tf.cosh(summed_pdfs)))
            pdf_residual = tf.multiply(mean_log_pdf, -1)

            # PRIOR AGAINST LARGE STANDARD DEVIATIONS
            # We sum all of the standard deviations and bias them towards being smaller in a knock-off of a Jeffreys log
            # uniform prior.
            mean_sigma_per_object = tf.reduce_mean(coefficients['std_deviations'], axis=1, keepdims=False)
            sigma_residual = tf.multiply(tf.reduce_mean(tf.log(tf.cosh(mean_sigma_per_object))), self.std_deviation_strength)

            # PRIOR AGAINST MEANS OUTSIDE OF THE REDSHIFT RANGE
            # We check for any means outside the range and massively scale the loss function if they're there.
            #little_means = tf.less(coefficients['means'], self.redshift_range[0])
            #big_means = tf.greater(coefficients['means'], self.redshift_range[1])

            # Add together counts, multiply by 10 to make it bigger
            #bad_means_multiplier = tf.multiply(10., tf.add(tf.count_nonzero(little_means, dtype=tf.float32),
            #                                               tf.count_nonzero(big_means, dtype=tf.float32)))

            # ADD LOG MEANS AND MULTIPLY BY THE OUTSIDE OF REDSHIFT RANGE MULTIPLIER
            total_residual = tf.add(sigma_residual, tf.add(cdf_residual, pdf_residual))

            return {'total_residual': total_residual,
                    'sigma_residual': sigma_residual,
                    'cdf_residual': cdf_residual,
                    'pdf_residual': pdf_residual}


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

        # Sum each mixture if requested, giving the cdf of the overall mixture density for that object
        if sum_mixtures:
            return np.sum(all_the_pdfs, axis=0)
        else:
            return all_the_pdfs

    @staticmethod
    def cdf_multiple_points(x_data, coefficients: dict, sum_mixtures=True):
        """Lossfunc defined in simple numpy arrays. Will instead return a cdf for the given object. Fastest at
        evaluating lots of x data points: for instance, for initial minimum finding or for plotting a cdf.

            Args:
                x_data: the y/dependent variable data to evaluate against. Can be an array or a float.
                coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                              of those values for the given object.
                sum_mixtures: defines whether or not we should sum the mixture distribution at each point or return
                              a big array of all the mixture components individually.

            Returns:
                A 1D array of the cdf value(s) at whatever point(s) you've evaluated it at.
        """
        # Typecast x_data as a 1D numpy array and work out how many mixtures there are
        x_data = np.array([x_data]).flatten()
        n_mixtures = coefficients['weights'].size
        all_the_cdfs = np.empty((n_mixtures, x_data.size))

        # Cycle over each mixture and evaluate the pdf
        i = 0
        for a_weight, a_mean, a_std_d in zip(coefficients['weights'], coefficients['means'],
                                             coefficients['std_deviations']):
            all_the_cdfs[i, :] = scipy_normal.cdf(x_data, loc=a_mean, scale=a_std_d) * a_weight
            i += 1

        # Sum each mixture if requested, giving the cdf of the overall mixture density for that object
        if sum_mixtures:
            return np.sum(all_the_cdfs, axis=0)
        else:
            return all_the_cdfs

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
    def cdf_single_point(x_point: Union[float, int], coefficients: dict):
        """CDF of lossfunc implemented in a much faster way for single point evaluations.
        Args:
            x_point: the y/dependent variable data to evaluate against. Can only be a float or int.
            coefficients: a dictionary with a 'weights', 'alpha' and 'beta' argument, each pointing to 1D arrays
                          of those values for the given object.

        Returns:
            A float of the cdf evaluated at a single point.
        """
        return np.sum(scipy_normal.cdf(x_point, loc=coefficients['means'], scale=coefficients['std_deviations'])
                      * coefficients['weights'])

    def draw_random_variables(self, n_samples: int, coefficients: dict):
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

        # Define an array of samples and work out how many we'll want to draw from each mixture. We ensure that the
        # array of samples has a value that is out of the allowed redshift range so we can work out which ones still
        # need editing.
        random_variates = np.zeros(n_samples) - self.redshift_range[0] - 1.
        number_weights = np.around(coefficients['weights'] * n_samples).astype(int)

        # Add 0 to the start of number_weights so that when writing random variates we start at the beginning of
        # the array
        number_weights = np.append([0], number_weights)

        # If number_weights sums to more than the number of points specified (due to a rounding error) then subtract the
        # rounding error (not normally more than just 1) from the largest weights.
        difference = n_samples - np.sum(number_weights)

        if difference != 0:
            number_weights[np.argmax(number_weights)] += difference

        # Make a cumulative sum so that we've got more by way of
        number_indices = np.cumsum(number_weights)

        # Draw some random variables for fun
        i = 1
        for a_mean, a_std_d in zip(coefficients['means'], coefficients['std_deviations']):

            # Cycle over drawing random variates until they're all in the correct range
            variates_still_to_draw = np.array([1])
            while variates_still_to_draw.size != 0:
                variates_still_to_draw = np.where(np.logical_or(
                    random_variates[number_indices[i-1]:number_indices[i]] < self.redshift_range[0],
                    random_variates[number_indices[i-1]:number_indices[i]] > self.redshift_range[1]))[0]

                random_variates[number_indices[i-1]:number_indices[i]][variates_still_to_draw] = \
                    scipy_normal.rvs(loc=a_mean, scale=a_std_d, size=variates_still_to_draw.size)

            i += 1

        return random_variates

# todo: only normal_cdf has been updated to have redshift_range etc properly defined and to integrate properly.
