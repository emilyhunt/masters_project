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


def where_valid_redshifts(spectroscopic_z, photometric_z, validity_condition: str='greater_than_zero'):
    """Quick function to find and return where valid values exist for plotting.

    Args:
        spectroscopic_z (array-like): true/spec redshifts.
        photometric_z (array-like): inferred/phot redshifts.
        validity_condition (str): method to use. Choose from:
            'greater_than_zero' - checks for negative redshifts (e.g. -99 in the CANDELS catalogue.)

    Returns:
        An array of indices of valid redshifts.
    """
    # Work out where there are valid spec and phot values
    # > 0 is valid for the CANDELS catalogue, as anything set to -99 is invalid.
    if validity_condition is 'greater_than_zero':
        valid = np.where(np.logical_and(np.asarray(spectroscopic_z) > 0, np.asarray(photometric_z) > 0))[0]

    # Otherwise... the specified validity condition isn't implemented!
    else:
        raise ValueError('specified validity condition is not implemented!')

    return valid


def calculate_nmad(spectroscopic_z, photometric_z, validity_condition: str='greater_than_zero'):
    """Calculates the normalised median absolute deviation between a set of photometric and spectroscopic redshifts,
    which is defined as:

        NMAD = 1.4826 * median( absolute( (z_phot - z_spec) / (1 + z_phot) ) )

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


def make_3dhst_photometry_table(hdu_table, keys_to_keep, new_key_names=None):  # todo: doesn't make sense that this isn't like read_save. Should change to be like that instead.
    """Creates a new photometry table in the pandas framework given a set of keys to keep and new names for them. By
    creating a new object, it's nicer and also not read-only. You should delete the hdu_table after calling this if you
    don't need it anymore.

    Args:
        hdu_table (astropy BinTableHDU object): table to read in.
        keys_to_keep (list of str): keys to bother keeping from the read table.
        new_key_names (None or list of str): new key names to assign to elements.

    Returns:
        A list, containing:
            0. a scrumptuous and lovely pandas data frame.
            1. a list of all the keys corresponding to flux (aka feed for a neural net.)
            2. a list of all the keys corresponding to error (aka more feed for a neural net.)

    """
    # Step 1: let's make a data frame
    data = pd.DataFrame()

    if new_key_names is None:
        new_key_names = keys_to_keep

    for new_key, old_key in zip(new_key_names, keys_to_keep):
        data[new_key] = hdu_table.data[old_key].byteswap().newbyteorder()

    # Step 2: make fresh lists of the flux and error keys
    flux_list = []
    error_list = []

    for a_string in new_key_names:

        if 'f_' in a_string:
            flux_list.append(a_string)

        elif 'e_' in a_string and a_string != 'use_phot':
            error_list.append(a_string)

    return [data, flux_list, error_list]


def check_photometric_coverage(data, columns_to_check: list, coverage_minimum: float=0.95, verbose: bool=True,
                               check: str='not_minus_99', valid_photometry_column: str='use_phot',
                               star_column: str='star_flag'):
    """Looks at which bands have enough coverage to be worth using, and suggests rows and columns to drop to achieve
    this.

    Args:
        data (pd.DataFrame): data frame to act on.
        columns_to_check (list of str): columns in the data frame to check.
        coverage_minimum (float): required amount of coverage in a column to keep it.
        verbose (bool): controls how much we print. True prints moaaar.
        check (str): check to perform for valid values. Currently only one, which is 'not_minus_99'.
        valid_photometry_column (str): name of a column that specifies rows that aren't for some reason screwed up.

    Returns:
        A list of:
            0. columns_to_keep that can be used as training data (yay.)
            1. columns_to_remove that have less than coverage_minimum
            2. rows that are stars
            3. rows that have bad photometry
            4. rows that have good photometry but are incomplete for these rows
    """
    # Have a look at how many good values are in all columns
    column_coverage = np.zeros(len(columns_to_check), dtype=np.float)
    columns_to_remove = []
    columns_to_keep = []
    total_length = data.shape[0]

    if check == 'not_minus_99':
        # Check for rows that are actually stars...
        rows_to_remove_are_stars = np.where(data[star_column] == 1)[0]
        rows_to_keep_are_stars = np.where(data[star_column] != 1)[0]

        # Check for rows that have a bad photometry flag
        rows_to_remove_bad_phot = np.where(data[valid_photometry_column].iloc[rows_to_keep_are_stars] != 1)[0]
        rows_to_keep_bad_phot = np.where(data[valid_photometry_column].iloc[rows_to_keep_are_stars] == 1)[0]  # todo: fuck, you're lazy...

        # Cycle over columns looking at their coverage
        for column_i, a_column in enumerate(columns_to_check):
            column_coverage[column_i] = \
                np.where(data[a_column].iloc[rows_to_keep_bad_phot] > -99.0)[0].size / total_length

            # Add the column to the naughty list if it isn't good enough
            if column_coverage[column_i] < coverage_minimum:
                columns_to_remove.append(a_column)
            else:
                columns_to_keep.append(a_column)

        # Now, check look for bad rows that still have missing data. We do a massive np.where check across the whole
        # array, then look for instances of True (where the condition isn't satisfied), then work out indeces of rows
        # with more than one offender.
        rows_to_remove_incomplete_phot = np.where(np.count_nonzero(
            np.where(data[columns_to_keep].iloc[rows_to_keep_bad_phot] > -99.0, False, True), axis=1) > 0)[0]

    else:
        raise ValueError('specified check not implemented.')

    if verbose:
        print('I have checked the coverage of the data. I found that:')
        print('{} of {} rows have a star flag and are not included.'
              .format(rows_to_remove_are_stars.size, total_length))
        print('{} of {} rows had a bad photometry warning flag and are not included.'
              .format(rows_to_remove_bad_phot.size, data[columns_to_keep].iloc[rows_to_keep_are_stars].shape[0]))
        print('{} out of {} columns do not have coverage over {}% on good sources.'
              .format(len(columns_to_remove), len(columns_to_check), coverage_minimum * 100))
        print('These were: {}'
              .format(columns_to_remove))
        print('I also found that {} of {} rows would still have invalid values even after removing all the above.'
              .format(rows_to_remove_incomplete_phot.size, data[columns_to_keep].iloc[rows_to_keep_bad_phot].shape[0]))
        print('This leaves a total of {:.2f}% of initial rows in the final data set.'
              .format(100 * (1 - (rows_to_remove_incomplete_phot.size + rows_to_remove_bad_phot.size
                                  + rows_to_remove_are_stars.size) / total_length)))

    return [columns_to_keep, columns_to_remove,
            rows_to_remove_are_stars, rows_to_remove_bad_phot, rows_to_remove_incomplete_phot]





