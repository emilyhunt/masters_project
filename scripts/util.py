"""A set of useful utilities for redshift calculations."""

import numpy as np
import pandas as pd
from scipy.io.idl import readsav as scipy_readsav
from scipy.optimize import curve_fit
from scipy.stats import norm
from typing import Optional


def read_save(target: str, columns_to_keep: Optional[list]=None, new_column_names: Optional[dict]=None):
    """Reads in a save file sensibly for use.

    Args:
        target (str): the location of the file to read in.
        columns_to_keep (list of strings): should be a list of columns, or None to keep all columns.
        new_column_names (dict): should be a dictionary with old names as keys and new names as values. Or None to not
            do any re-naming.

    Returns:
        The read in file as a pandas DataFrame.
    """
    # Read in the specified save file
    data = scipy_readsav(target, python_dict=True, verbose=False)  # Set verbose=True for more info on the file

    # Remove the 'readme' key from the dictionary, if it exists - else return None
    data.pop('readme', None)

    # Ensure the byte order of the read in numpy arrays is the same as on this machine. Solves an issue as described at:
    # https://pandas.pydata.org/pandas-docs/stable/gotchas.html#byte-ordering-issues
    for a_key in data.keys():
        data[a_key] = data[a_key].byteswap().newbyteorder('L')

    # Cast data as a DataFrame
    data = pd.DataFrame(data)

    # Rename any relevant columns
    if new_column_names is not None:
        data.rename(columns=new_column_names, inplace=True)

    # Collate the columns that we want and remove columns if the user has done anything with columns_to_keep
    if columns_to_keep is not None:
        # User can ask to only keep new columns
        if columns_to_keep is 'only_new_ones':
            data = data[new_column_names.values]
        # Or, we use a list
        else:
            data = data[columns_to_keep]

    return data


def calculate_nmad(spectroscopic_z, photometric_z, validity_condition: str='greater_than_zero'):
    """Calculates the normalised median absolute deviation between a set of photometric and spectroscopic redshifts,
    which is defined as:

        NMAD = 1.4826 * median( absolute( (z_phot - z_spec) / (1 + z_phot) ) )

    Args:
        spectroscopic_z (list-like): spectroscopic/correct redshifts.
        photometric_z (list-like): photometric/inferred redshifts.
        validity_condition (str):

    Returns:
        The NMAD, a float.
    """
    # Flatten arrays to avoid issues with them being the wrong size
    spectroscopic_z = spectroscopic_z.flatten()
    photometric_z = photometric_z.flatten()

    # Work out where there are valid spec and phot values
    # > 0 is valid for the CANDELS catalogue, as anything set to -99 is invalid.
    if validity_condition is 'greater_than_zero':
        valid = np.where(np.logical_and(spectroscopic_z > 0, photometric_z > 0))[0]
    else:
        raise ValueError('specified validity condition is not implemented!')

    # Calculate & return the NMAD
    return 1.4826 * np.median(np.abs((photometric_z[valid] - spectroscopic_z[valid]) / (1 + photometric_z[valid])))


def single_gaussian_to_fit(x, standard_deviation, A, mean=0):
    """Allows a Gaussian to be accessed to fit a curve to, providing some slightly simpler notation than
    scipy.stats.norm.pdf. Only fits Gaussians of mean zero.

    Args:
        x (np.ndarray or float): point(s) to evaluate against.
        standard_deviation (np.ndarray or float): standard deviations of Gaussians.
        A (int, float): the amplitude of the Gaussian.
    Returns:
        float or np.array of the pdf(s) that have been evaluated.

    """
    return A * norm.pdf(x, loc=0, scale=standard_deviation)


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


def fit_gaussians(x_range, y_range):
    """Function that handles fitting Gaussians to our final data using scipy.curve_fit. Is able to catch typical
    RuntimeErrors that occur from optimisation failing. Performs a fit for both a single and a double (convolved)
    Gaussian.

    Args:
        x_range (list-like): x points to fit against.
        y_range (list-like): y points to fit against.

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

    return my_params
