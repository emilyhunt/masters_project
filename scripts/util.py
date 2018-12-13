"""A set of useful utilities for redshift calculations."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm


def where_valid_redshifts(redshift_1, redshift_2=None, validity_condition: str='greater_than_zero'):
    """Quick function to find and return where valid values exist for plotting.

    Args:
        redshift_1 (array-like): first redshift to check
        redshift_2 (array-like or None): optional second redshifts to check. Both redshift 1 AND 2 must be valid to
            return array indices. Should have identical shape to redshift_1. Default value is None.
        validity_condition (str): method to use. Choose from:
            'greater_than_zero' - checks for negative redshifts (e.g. -99 in the CANDELS catalogue.)

    Returns:
        An array of indices of valid redshifts.
    """
    # Work out where there are valid spec and phot values
    # > 0 is valid for the CANDELS catalogue, as anything set to -99 is invalid.
    if validity_condition == 'greater_than_zero':
        if redshift_2 is not None:
            valid = np.where(np.logical_and(np.asarray(redshift_1) > 0, np.asarray(redshift_2) > 0))[0]
        else:
            valid = np.where(np.asarray(redshift_1) > 0)[0]

    elif validity_condition is None:
        return np.arange(0, redshift_1.shape[0])

    # Otherwise... the specified validity condition isn't implemented!
    else:
        raise ValueError('specified validity condition is not implemented!')

    return valid


def calculate_nmad(spectroscopic_z, photometric_z, validity_condition: str='greater_than_zero'):
    """Calculates the normalised median absolute deviation between a set of photometric and spectroscopic redshifts,
    which is defined as:

        NMAD = 1.4826 * median( absolute( (z_phot - z_spec - median(z_phot - z_spec)) / (1 + z_phot) ) )

    Args:
        spectroscopic_z (list-like): spectroscopic/correct redshifts.
        photometric_z (list-like): photometric/inferred redshifts.
        validity_condition (str): validity condition to use with where_valid_redshifts().

    Returns:
        The NMAD, a float.
    """
    # Flatten arrays to avoid issues with them being the wrong size
    spectroscopic_z = np.asarray(spectroscopic_z).flatten()
    photometric_z = np.asarray(photometric_z).flatten()

    # Work out where there are valid spec and phot values
    valid = where_valid_redshifts(spectroscopic_z, photometric_z, validity_condition=validity_condition)

    # Calculate & return the NMAD
    delta_z = photometric_z[valid] - spectroscopic_z[valid]
    median_delta_z = np.median(delta_z)

    return 1.4826 * np.median(np.abs((delta_z - median_delta_z) / (1 + spectroscopic_z[valid])))


def single_gaussian_to_fit(x, standard_deviation, A, mean=0):
    """Allows a Gaussian to be accessed to fit a curve to, providing some slightly simpler notation than
    scipy.stats.norm.pdf. Can fit Gaussians of any mean.

    Args:
        x (np.ndarray or float): point(s) to evaluate against.
        standard_deviation (np.ndarray or float): standard deviations of Gaussians.
        A (int, float): the amplitude of the Gaussian.
        mean (np.ndarray or float): mean of the Gaussian. Default is 0.
    Returns:
        float or np.array of the pdf(s) that have been evaluated.

    """
    return A * norm.pdf(x, loc=mean, scale=standard_deviation)


def single_normal_to_fit(x, standard_deviation, mean=0):
    """Allows a Normal distribution to be accessed to fit a curve to, providing some slightly simpler notation than
    scipy.stats.norm.pdf. Can fit normal distributions of any mean.

    Args:
        x (np.ndarray or float): point(s) to evaluate against.
        standard_deviation (np.ndarray or float): standard deviations of normals.
        mean (np.ndarray or float): mean of the normal. Default is 0.
    Returns:
        float or np.array of the pdf(s) that have been evaluated.

    """
    return norm.pdf(x, loc=mean, scale=standard_deviation)


def double_gaussian_to_fit(x, standard_deviation_1, standard_deviation_2, A, r):
    """Allows a double Gaussian to be accessed to fit a curve to, providing some slightly simpler notation than
    scipy.stats.norm.pdf. Only fits Gaussians of mean zero.

    Uses equation from Quadri+2010.

    Args:
        x (np.ndarray or float): point(s) to evaluate against.
        standard_deviation_1 (np.ndarray or float): standard deviations of first Gaussians.
        standard_deviation_2 (np.ndarray or float): standard deviations of second Gaussians.
        A (np.ndarray, float): the amplitude of the Gaussians.
        r (np.ndarray, float): the ratio of the Gaussians.

    Returns:
        float or np.array of the pdf(s) that have been evaluated.
    """
    sqrt_2 = np.sqrt(2)
    term_1 = A * norm.pdf(x, loc=0, scale=sqrt_2*standard_deviation_1)
    term_2 = A * 2 * r * norm.pdf(x, loc=0, scale=np.sqrt(standard_deviation_1**2 + standard_deviation_2**2))
    term_3 = A * r**2 * norm.pdf(x, loc=0, scale=sqrt_2 * standard_deviation_2)
    return term_1 + term_2 + term_3


def fit_gaussians(x_range, y_range, fit_double_gaussian=True):
    """Function that handles fitting Gaussians to our final data using scipy.curve_fit. Is able to catch typical
    RuntimeErrors that occur from optimisation failing. Performs a fit for both a single and a double (convolved)
    Gaussian.

    Args:
        x_range (list-like): x points to fit against.
        y_range (list-like): y points to fit against.
        fit_double_gaussian (bool): whether or not to try and fit a double Gaussian.

    Returns:
        A dict of parameters of fitting Gaussians, both for single and double Gaussians, which has keys:
            s_s = standard deviation of single Gaussian fit
            s_A = amplitude of single Gaussian fit
            d_s1 = standard deviation of first Gaussian in double fit
            d_s2 = standard deviation of second Gaussian in double fit
            d_A = amplitude of double Gaussian fit
            d_r = ratio between the two Gaussians.

        If the method fails in either case, then parameters for a flat curve are returned.
    """
    # Make a blank dictionary for keeping our params in
    my_params = {}

    # Fit the first Gaussian
    try:
        params_optimized, params_covariance = curve_fit(single_gaussian_to_fit, x_range, y_range,
                                                        p0=[1, 1],
                                                        bounds=([0.01, 0],
                                                                [np.inf, np.inf]),
                                                        verbose=0, method='dogbox')
        my_params['s_s'] = params_optimized[0]
        my_params['s_A'] = params_optimized[1]
    except RuntimeError:
        print('Unable to fit single Gaussian, likely due to maximum number of function evals being exceeded!')
        my_params['s_s'] = 1
        my_params['s_A'] = 0

    # Fit the double Gaussian
    if fit_double_gaussian:
        try:
            params_optimized, params_covariance = curve_fit(double_gaussian_to_fit, x_range, y_range,
                                                            p0=[1, 2, 1, 0.5],
                                                            bounds=([0.01, 0.01, 0, 0],
                                                                    [np.inf, np.inf, np.inf, np.inf]),
                                                            verbose=0, method='dogbox')
            my_params['d_s1'] = params_optimized[0]
            my_params['d_s2'] = params_optimized[1]
            my_params['d_A'] = params_optimized[2]
            my_params['d_r'] = params_optimized[3]
        except RuntimeError:
            print('Unable to fit double Gaussian, likely due to maximum number of function evals being exceeded!')
            my_params['d_s1'] = 1
            my_params['d_s2'] = 1
            my_params['d_A'] = 0
            my_params['d_r'] = 0.5
    else:
        my_params['d_s1'] = 1
        my_params['d_s2'] = 1
        my_params['d_A'] = 0
        my_params['d_r'] = 0.5

    return my_params


def fit_normal(x_range, y_range):
    """Function that handles fitting a normal distribution to our data using scipy.curve_fit. Is able to catch typical
    RuntimeErrors that occur from optimisation failing. Performs a fit for only a single Normal distro.

    Args:
        x_range (list-like): x points to fit against.
        y_range (list-like): y points to fit against.

    Returns:
        A dict of parameters of fit Normal parameters, which has keys:
            s_m = mean of single Normal fit
            s_s = standard deviation of single Normal fit

        If the method fails, then parameters for a flat curve are returned.
    """
    # Make a blank dictionary for keeping our params in
    my_params = {}

    # Fit the first Normal
    try:
        params_optimized, params_covariance = curve_fit(single_normal_to_fit, x_range, y_range,
                                                        p0=[1, 1],
                                                        bounds=([0.00000001, -np.inf],
                                                                [np.inf, np.inf]),
                                                        verbose=0, method='dogbox')
        my_params['s_s'] = params_optimized[0]
        my_params['s_m'] = params_optimized[1]
    except RuntimeError:
        print('Unable to fit single Normal, likely due to maximum number of function evals being exceeded!')
        my_params['s_s'] = 1
        my_params['s_m'] = 0

    return my_params


