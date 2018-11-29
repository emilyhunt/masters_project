"""A set of useful utilities for redshift calculations."""

import numpy as np
import pandas as pd
from astropy.io import fits
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
    if validity_condition is 'greater_than_zero':
        if redshift_2 is not None:
            valid = np.where(np.logical_and(np.asarray(redshift_1) > 0, np.asarray(redshift_2) > 0))[0]
        else:
            valid = np.where(np.asarray(redshift_1) > 0)[0]

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


def read_fits(fits_file_location: str, keys_to_keep: list, fits_index_to_read: int=0, new_key_names: bool=None,
              get_flux_and_error_keys: bool=False):
    """Creates a new photometry table in the pandas framework given a fits file, a set of keys to keep and new names for 
    them. By creating a new object, it's nicer and also not read-only.

    Args:
        fits_file_location (location of an astropy BinTableHDU object): table to read in.
        keys_to_keep (list of str): keys to bother keeping from the read table.
        fits_index_to_read (int): index of the fits file to read in. Default is 1 (0 is usually a header.)
        new_key_names (None or list of str): new key names to assign to elements.
        get_flux_and_error_keys (bool): whether or not to look for flux and error keys and return them.

    Returns:
        A list, containing:
            0. a scrumptuous and lovely pandas data frame.
            1. a list of all the keys corresponding to flux (aka feed for a neural net.)
            2. a list of all the keys corresponding to error (aka more feed for a neural net.)

    """
    # Step 0: read in the fits file
    fits_file = fits.open(fits_file_location)[fits_index_to_read]

    # Step 1: let's make a data frame
    data = pd.DataFrame()

    if new_key_names is None:
        new_key_names = keys_to_keep

    # We cycle over the fits file, also doing byte shit as they seem to be wrong otherwise
    for new_key, old_key in zip(new_key_names, keys_to_keep):
        data[new_key] = fits_file.data[old_key].byteswap().newbyteorder()

    # Step 2: make fresh lists of the flux and error keys, if requested
    if get_flux_and_error_keys:
        flux_list = []
        error_list = []

        for a_string in new_key_names:

            if 'f_' in a_string:
                flux_list.append(a_string)

            elif 'e_' in a_string and a_string != 'use_phot':
                error_list.append(a_string)

        return [data, flux_list, error_list]

    # Or, just return data
    else:
        return data


def check_photometric_coverage_3dhst(data, flux_keys: list, error_keys: list, coverage_minimum: float=0.95, verbose: bool=True,
                                     valid_photometry_column: str='use_phot', z_spec_column: str='z_spec',
                                     missing_flux_handling: Optional[str]=None,
                                     missing_error_handling: Optional[str]=None):
    """Looks at which bands have enough coverage to be worth using, and suggests rows and columns to drop to achieve
    this. ONLY COMPATIBLE WITH 3DHST DATA, as other surveys may mark poor data in different ways.

    Args:
        data (pd.DataFrame): data frame to act on.
        columns_to_check (list of str): columns in the data frame to check.
        coverage_minimum (float): required amount of coverage in a column to keep it.
        verbose (bool): controls how much we print. True prints moaaar.
        valid_photometry_column (str): name of a column that specifies rows that aren't for some reason screwed up.
        z_spec_column (str): name of the column of spectroscopic redshifts.
        missing_flux_handling (None or str): method to use to replace missing flux points.
        missing_error_handling (None or str): method to use to replace missing error points.

    Returns:
        A list containing data that has valid spectroscopic redshifts (0th element) and data without valid
        spectroscopic redshifts (1st element).
    """
    # Have a look at how many good values are in all columns
    all_columns_to_check = flux_keys + error_keys
    column_coverage = np.zeros(len(all_columns_to_check), dtype=np.float)
    columns_to_remove = []
    original_length = data.shape[0]

    # Check everything, by looking at:
    # 1. Whether or not they're affected by poor photometry/are stars (use_phot does both of these things)
    # 2. Removing columns with the poorest coverage
    # 3. Removing or processing rows that still have missing photometric data points
    # 4. Whether or not they have spectroscopic redshifts available

    print('Checking the photometric coverage of the dataset...')

    # Step 1: So! The photometry check
    bad_photometry = np.where(data[valid_photometry_column] != 1)[0]

    print('Removing {} of {} galaxies with poor photometry.'
          .format(bad_photometry.size, data.shape[0]))

    data = data.drop(index=bad_photometry).reset_index(drop=True)

    # Step 2: Remove any columns with very poor coverage, as they aren't worth our time potentially
    # Cycle over columns looking at their coverage: first for fluxes, then almost the same again for errors (except we
    # only record the column coverages once, for flux keys)
    for column_i, a_column in enumerate(flux_keys):
        column_coverage[column_i] = \
            np.where(data[a_column] > -99.0)[0].size / data.shape[0]

        # Add the column to the naughty list if it isn't good enough
        if column_coverage[column_i] < coverage_minimum:
            columns_to_remove.append(a_column)
        flux_keys.remove(a_column)

    for column_i, a_column in enumerate(error_keys):
        a_column_coverage = np.where(data[a_column] > -99.0)[0].size / data.shape[0]

        # Add the column to the naughty list if it isn't good enough
        if a_column_coverage < coverage_minimum:
            columns_to_remove.append(a_column)
        error_keys.remove(a_column)

    print('{} out of {} columns do not have coverage over {}% on good sources.'
          .format(len(columns_to_remove), len(all_columns_to_check), coverage_minimum * 100))
    print('These were: {}'
          .format(columns_to_remove))

    data = data.drop(columns=columns_to_remove).reset_index(drop=True)

    # Step 3: Deal with missing photometric data points in one of a number of ways

    # Method one: make them go away. We count occurrences of invalid redshifts row-wise, and then drop said bad rows
    if missing_flux_handling is None:
        print('Dealing with missing fluxes by removing said rows...')
        rows_to_drop = np.where(np.count_nonzero(np.where(data[flux_keys] > -99.0,
                                                          False, True), axis=1) > 0)[0]
        data = data.drop(index=rows_to_drop).reset_index(drop=True)
        print('Removed {} of {} galaxies.'.format(rows_to_drop.size, data.shape[0]))

    # Method two: set missing points to the mean value of that row.
    elif missing_flux_handling == 'row_mean':
        print('Dealing with missing data by setting it to the mean of rows...')
        # Take the mean of each row, then reshape it into a vertical array and make it horizontally as wide as the
        # number of flux rows
        row_means = np.mean(np.asarray(data[flux_keys]), axis=1)
        row_means = np.repeat(row_means.reshape(-1, 1), len(flux_keys), axis=1)

        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99.0, False, True)
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=row_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    # Method three: set missing points to the mean value of that column.
    elif missing_flux_handling == 'column_mean':
        print('Dealing with missing data by setting it to the mean of columns...')
        # Take the mean of each column, then stretch it to be as long as the data itself
        column_means = np.mean(np.asarray(data[flux_keys]), axis=0)
        column_means = np.repeat(column_means.reshape(1, -1), data.shape[0], axis=0)

        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99.0, False, True)
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=column_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    else:
        raise ValueError('specified missing_flux_handling not found/implemented/supported.')

    # Now, do the same but with errors!
    if missing_error_handling is None:
        print('Dealing with missing errors by removing said rows...')
        rows_to_drop = np.where(np.count_nonzero(np.where(data[error_keys] > -99.0,
                                                          False, True), axis=1) > 0)[0]
        data = data.drop(index=rows_to_drop).reset_index(drop=True)
        print('Removed {} of {} galaxies.'.format(rows_to_drop.size, data.shape[0]))

    elif missing_error_handling == 'big_value':
        print('Dealing with missing errors by setting them to a large value...')
        # Set it to 100 times the maximum error in the catalogue
        the_big_number = np.max(np.asarray(data[error_keys])) * 100

        # Grab the fluxes that need fixing and make the fixeyness happen!
        errors_to_fix = np.where(data[error_keys] == -99.0, False, True)
        data[error_keys] = data[error_keys].where(errors_to_fix, other=the_big_number)
        print('{} of {} errors were modified to a value of {}.'
              .format(errors_to_fix.size - np.count_nonzero(errors_to_fix), errors_to_fix.size, the_big_number))

    else:
        raise ValueError('specified missing_error_handling not found/implemented/supported.')

    # Step 4: Split the data into sets with and without spectroscopic redshifts
    has_spec_z = np.where(np.asarray(data[z_spec_column]) != -99.0)[0]
    data_spec_z = data.iloc[has_spec_z].reset_index(drop=True)
    data_no_spec_z = data.drop(index=has_spec_z).reset_index(drop=True)
    print('{} out of {} galaxies have spectroscopic redshifts.'.format(has_spec_z.size, data.shape[0]))

    return [data_spec_z, data_no_spec_z, flux_keys, error_keys]


"""

    if check == 'not_minus_99':
        # Reduce to everything that's got spectroscopic redshifts
        rows_to_remove_no_spec = np.where(data)

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
            
    """





