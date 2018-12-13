"""All functions for preprocessing input photometric data."""

import numpy as np
import pandas as pd
import sys
from typing import Optional, Union


def missing_data_handler(data, flux_keys: list, error_keys: list,
                         band_central_wavelengths: list, coverage_minimum: float=0.95,
                         valid_photometry_column: str='use_phot', z_spec_column: str='z_spec',
                         z_grism_column: str='z_grism',
                         z_grism_error_columns: Union[tuple, list]=('z_grism_l68', 'z_grism_u68'),
                         z_grism_minimum_snr: Optional[float]=None,
                         missing_flux_handling: Optional[str]=None,
                         missing_error_handling: Optional[str]=None):
    """Looks at which bands have enough coverage to be worth using, and suggests rows and columns to drop to achieve
    this. ONLY COMPATIBLE WITH 3DHST DATA, as other surveys may mark poor data in different ways.

    Args:
        data (pd.DataFrame): data frame to act on.
        flux_keys (list of str): flux keys in the data frame to check.
        error_keys (list of str): error keys in the data frame to check.
        band_central_wavelengths (list of floats): central wavelengths of the filter bands.
        coverage_minimum (float): required amount of coverage in a column to keep it.
        valid_photometry_column (str): name of a column that specifies rows that aren't for some reason screwed up.
        z_spec_column (str): name of the column of spectroscopic redshifts.
        z_grism_column (str): name of the column of grism redshifts.
        z_grism_minimum_snr (None or float): minimum signal to noise ratio to use when selecting which grism redshifts
            to use.
        z_grism_error_columns (tuple or list): names of the lower and upper bound z_grism errors.
        missing_flux_handling (None or str): method to use to replace missing flux points.
        missing_error_handling (None or str): method to use to replace missing error points.

    Returns:
        A list containing data that has valid spectroscopic redshifts (0th element) and data without valid
        spectroscopic redshifts (1st element).
    """
    # Make sure we're using local copies of the flux and error keys
    flux_keys = flux_keys.copy()
    error_keys = error_keys.copy()
    band_central_wavelengths = band_central_wavelengths.copy()

    # Have a look at how many good values are in all columns
    all_columns_to_check = flux_keys + error_keys
    column_coverage = np.zeros(len(all_columns_to_check), dtype=np.float)
    flux_columns_to_remove = []
    error_columns_to_remove = []
    bands_to_remove = []

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
    for band_i, (a_flux_key, a_error_key) in enumerate(zip(flux_keys, error_keys)):
        column_coverage[band_i] = \
            np.where(data[a_flux_key] > -99.0)[0].size / data.shape[0]

        # Add the column to the naughty list if it isn't good enough
        if column_coverage[band_i] < coverage_minimum:
            flux_columns_to_remove.append(a_flux_key)
            error_columns_to_remove.append(a_error_key)
            bands_to_remove.append(band_i)

    band_central_wavelengths = np.delete(np.asarray(band_central_wavelengths), bands_to_remove)

    print('{} out of {} columns do not have coverage over {}% on good sources.'
          .format(len(flux_columns_to_remove), len(flux_keys), coverage_minimum * 100))
    print('These were: {}'
          .format(flux_columns_to_remove))
    print('We also remove the error columns: {}'
          .format(error_columns_to_remove))

    # Actually do the removing from the key lists
    for a_flux_key, a_error_key in zip(flux_columns_to_remove, error_columns_to_remove):
        flux_keys.remove(a_flux_key)
        error_keys.remove(a_error_key)

    data = data.drop(columns=flux_columns_to_remove + error_columns_to_remove).reset_index(drop=True)

    # Step 3: Deal with missing photometric data points in one of a number of ways

    # Method one: make them go away. We count occurrences of invalid redshifts row-wise, and then drop said bad rows
    if missing_flux_handling is None:
        print('Dealing with missing fluxes by removing said rows...')
        rows_to_drop = np.where(np.count_nonzero(np.where(data[flux_keys] > -99.0,
                                                          False, True), axis=1) > 0)[0]
        data = data.drop(index=rows_to_drop).reset_index(drop=True)

        # Add a column of zeros to track how many points are inferred in each row
        data['inferred_points'] = np.zeros(data.shape[0], dtype=int)

        print('Removed {} of {} galaxies.'.format(rows_to_drop.size, data.shape[0]))

    # Method two: set missing points to the mean value of that row.
    elif missing_flux_handling == 'row_mean':
        print('Dealing with missing data by setting it to the mean of rows...')
        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99.0, False, True)

        # Set bad values to np.nan before taking means so the mean ignores them
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=np.nan)

        # Take the mean of each row, then reshape it into a vertical array and make it horizontally as wide as the
        # number of flux rows
        row_means = np.nanmean(np.asarray(data[flux_keys]), axis=1)
        row_means = np.repeat(row_means.reshape(-1, 1), len(flux_keys), axis=1)

        # Record how much each row has been messed with
        data['inferred_points'] = np.sum(np.invert(fluxes_to_fix), axis=1, dtype=np.int)

        # Assign the new values
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=row_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    # Method three: set missing points to the mean value of that column.
    elif missing_flux_handling == 'column_mean':
        print('Dealing with missing data by setting it to the mean of columns...')
        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99.0, False, True)

        # Set bad values to np.nan before taking means so the mean ignores them
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=np.nan)

        # Take the mean of each column (nanmean ignores bad values), then stretch it to be as long as the data itself
        column_means = np.nanmean(np.asarray(data[flux_keys]), axis=0)
        column_means = np.repeat(column_means.reshape(1, -1), data.shape[0], axis=0)

        # Record how much each row has been messed with
        data['inferred_points'] = np.sum(np.invert(fluxes_to_fix), axis=1, dtype=np.int)

        # Assign the new values
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=column_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    # Method four: set missing points to the mean value of that column, but normalised with the help of row means.
    elif missing_flux_handling == 'normalised_column_mean':
        print('Dealing with missing data by setting it to the row-normalised mean of columns...')
        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99.0, False, True)

        # Set bad values to np.nan before taking means so the mean ignores them
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=np.nan)

        # Take the mean of each column, then stretch it to be as long as the data itself (and grab the overall mean too)
        column_means = np.nanmean(np.asarray(data[flux_keys]), axis=0)
        overall_mean = np.nanmean(column_means)
        column_means = np.repeat(column_means.reshape(1, -1), data.shape[0], axis=0)

        # Take the mean of each row, then reshape it into a vertical array and make it horizontally as wide as the
        # number of flux rows
        row_means = np.nanmean(np.asarray(data[flux_keys]), axis=1)
        row_means = np.repeat(row_means.reshape(-1, 1), len(flux_keys), axis=1)

        normalised_means = column_means + row_means - overall_mean

        # Record how much each row has been messed with
        data['inferred_points'] = np.sum(np.invert(fluxes_to_fix), axis=1, dtype=np.int)

        # Assign the new values
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=normalised_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    # Method five: set missing points to the mean value of that column, but normalised with the help of row mean ratios.
    elif missing_flux_handling == 'normalised_column_mean_ratio':
        print('Dealing with missing data by setting it to the ratio row-normalised mean of columns...')
        # Grab the fluxes that need fixing and make the fixeyness happen!
        fluxes_to_fix = np.where(data[flux_keys] == -99., False, True)

        # Set bad values to np.nan before taking means so the mean ignores them
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=np.nan)

        # Take the mean of each column, then stretch it to be as long as the data itself (and grab the overall mean too)
        column_means = np.nanmean(np.asarray(data[flux_keys]), axis=0)
        overall_mean = np.nanmean(np.asarray(data[flux_keys]))
        column_means = np.repeat(column_means.reshape(1, -1), data.shape[0], axis=0)

        # Take the mean of each row, then reshape it into a vertical array and make it horizontally as wide as the
        # number of flux rows
        row_means = np.nanmean(np.asarray(data[flux_keys]), axis=1)
        row_means = np.repeat(row_means.reshape(-1, 1), len(flux_keys), axis=1)

        normalised_means = column_means * row_means / overall_mean

        # Record how much each row has been messed with
        data['inferred_points'] = np.sum(np.invert(fluxes_to_fix), axis=1, dtype=np.int)

        # Assign the new values
        data[flux_keys] = data[flux_keys].where(fluxes_to_fix, other=normalised_means)
        print('{} of {} fluxes were modified.'
              .format(fluxes_to_fix.size - np.count_nonzero(fluxes_to_fix), fluxes_to_fix.size))

    # Method six: interpolate linearly between neighbouring bands.
    elif missing_flux_handling == 'linear_interpolation':
        if band_central_wavelengths is None:
            raise ValueError('interpolation activated yet no band central wavelengths were provided!')

        print('Dealing with missing data by interpolating between points...')

        changed_fluxes = 0
        rows_to_drop = np.array([])

        # Add a column of zeros to track how many points are inferred in each row
        data['inferred_points'] = np.zeros(data.shape[0], dtype=int)

        # Loop over all the different flux keys and interpolate
        for band_i, a_flux_key in enumerate(flux_keys):

            sys.stdout.write('\rInterpolating on band {}...'.format(band_i))
            sys.stdout.flush()

            # Correct for if we're at the start and work out the next and previous bands
            if band_i == 0:
                next_band = 2
                prev_band = 1
            elif band_i == len(flux_keys) - 1:
                next_band = band_i - 1
                prev_band = band_i - 2
            else:
                next_band = band_i + 1
                prev_band = band_i - 1

            # Grab some bits and bobs
            next_flux_key = flux_keys[next_band]
            prev_flux_key = flux_keys[prev_band]
            a_band_wave = band_central_wavelengths[band_i]
            next_band_wave = band_central_wavelengths[next_band]
            prev_band_wave = band_central_wavelengths[prev_band]

            # Work out which fluxes we need to fix
            fluxes_to_fix = np.where(data[a_flux_key] == -99.0, False, True)
            new_flux_truths = np.invert(fluxes_to_fix)
            changed_fluxes += np.count_nonzero(new_flux_truths)

            # Do linear y=mx + c interpolation on each point where necessary and store the result
            new_flux_values = np.where(new_flux_truths,
                                       (data[next_flux_key] - data[prev_flux_key]) / (next_band_wave - prev_band_wave)
                                       * a_band_wave + data[next_flux_key]
                                       - (data[next_flux_key] - data[prev_flux_key]) / (next_band_wave - prev_band_wave)
                                       * next_band_wave, 0.0)
            data[a_flux_key] = data[a_flux_key].where(fluxes_to_fix, other=new_flux_values)

            # Find and later remove any rows that have two consecutive missing values
            rows_to_drop = np.append(rows_to_drop, np.where(
                np.logical_or(np.logical_and(data[next_flux_key] == -99.0, new_flux_truths),
                              np.logical_and(data[prev_flux_key] == -99.0, new_flux_truths))))[0]

            # Add a column of zeros to track how many points are inferred in each row
            data['inferred_points'] += np.invert(fluxes_to_fix)

        old_row_number = data.shape[0]
        rows_to_drop = np.unique(rows_to_drop)
        data = data.drop(index=rows_to_drop).reset_index(drop=True)

        print('\n{} of {} fluxes were modified.'.format(changed_fluxes, data.shape[0] * len(flux_keys)))
        print('{} of {} rows had to be removed due to consecutive disallowed values.'
              .format(rows_to_drop.size, old_row_number))

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
        # Grab the errors that need fixing and make the fixeyness happen!
        errors_to_fix = np.where(data[flux_keys] == -99., False, True)

        # Set bad values to np.nan before taking means so the mean ignores them
        data[flux_keys] = data[flux_keys].where(errors_to_fix, other=np.nan)

        # Set it to 1000 times the maximum error in the catalogue
        the_big_number = np.nanmax(np.asarray(data[error_keys])) * 1000

        # Grab the fluxes that need fixing and make the fixeyness happen!
        data[error_keys] = data[error_keys].where(errors_to_fix, other=the_big_number)
        print('{} of {} errors were modified to a value of {}.'
              .format(errors_to_fix.size - np.count_nonzero(errors_to_fix), errors_to_fix.size, the_big_number))

    elif 'sigma_column' in missing_error_handling:
        # Get the sigma level from the string (which could be e.g. "5_sigma_column")
        sigma_level = float(missing_error_handling[:-13])
        print('Dealing with missing errors by setting them to {} times the median error in that band...'
              .format(sigma_level))

        # Grab the errors that need fixing and make the fixeyness happen!
        errors_to_fix = np.where(data[flux_keys] == -99., False, True)

        # Set bad values to np.nan before taking means so the mean ignores them (we also reshape so pandas.where works)
        data[flux_keys] = data[flux_keys].where(errors_to_fix, other=np.nan)
        column_means = np.nanmedian(np.asarray(data[error_keys]), axis=0) * sigma_level
        column_means = np.repeat(column_means.reshape(1, -1), data.shape[0], axis=0)

        # Grab the fluxes that need fixing and make the fixeyness happen!
        errors_to_fix = np.where(data[error_keys] == -99.0, False, True)
        data[error_keys] = data[error_keys].where(errors_to_fix, other=column_means)
        print('{} of {} errors were modified'
              .format(errors_to_fix.size - np.count_nonzero(errors_to_fix), errors_to_fix.size))

    else:
        raise ValueError('specified missing_error_handling not found/implemented/supported.')

    # Step 4: Split the data into sets with and without spectroscopic redshifts, and also include grism redshifts if
    # requested
    has_spec_z = np.where(np.asarray(data[z_spec_column]) != -99.0)[0]

    # If requested, find the
    if z_grism_minimum_snr is not None:
        # todo: add zgrism for training support
        raise NotImplementedError('z_grism support hasn\'t been written yet!')
        z_grism = np.asarray(data[z_grism_column])
        z_grism_u = 0
        z_grism_snr = z_grism
        has_grism_z = np.where(np.logical_and(z_grism != -1.0, True))

    data_spec_z = data.iloc[has_spec_z].copy().reset_index(drop=True)
    data_no_spec_z = data.drop(index=has_spec_z).copy().reset_index(drop=True)
    print('{} out of {} galaxies have spectroscopic redshifts.'.format(has_spec_z.size, data.shape[0]))

    # Step 5: if requested, also augment the dataset with grism redshifts

    return [data_spec_z, data_no_spec_z, flux_keys, error_keys]


class PhotometryScaler:

    def __init__(self, all_input_data, flux_keys, error_keys):
        """The photometry scaler handles conversion of the input photometry into a space that's easier to train with.
        This is implemented as a class, because identical transforms (such as adding the minimum and taking a log, etc)
        would need to be applied to all data, and this isn't possible without having a class around to keep track of
        certain key values.

        Args:
            all_input_data (list of pd.DataFrame): all data to be trained, tested or validated with, allowing us to grab
                certain important values for the whole data set.
            flux_keys (list of str): all fluxes that can be found in said input data.
            error_keys (list of str): all flux errors that can be found in said input data.
        """
        # Set values to inf so the first run will always store new ones, and store everything in a data frame
        self.photometry_stats = pd.DataFrame(np.inf * np.ones((len(flux_keys), 5)),
                                             columns=['minimum_flux', 'maximum_flux', 'minimum_sn', 'maximum_sn',
                                                      'mean_error'],
                                             index=flux_keys)

        first_run = True
        dict_of_all_errors = {}

        # We loop over all data frames in all_input_data to find the most extreme point
        for a_data_frame_i, a_data_frame in enumerate(all_input_data):
            n_objects_this_frame = a_data_frame.shape[0]
            for a_flux_key, a_error_key in zip(flux_keys, error_keys):
                # Firstly, let's record the absolute smallest signal to noise ratio (it may well be negative)
                fluxes = a_data_frame[a_flux_key].values
                errors = a_data_frame[a_error_key].values

                # Check we haven't been passed any incorrect values (that would fuck everything right up lol)
                bad_fluxes = np.count_nonzero(np.where(fluxes == -99., True, False))
                if bad_fluxes > 0:
                    raise ValueError('some fluxes were invalid!')

                # Set any errors that are zero to np.nan, then compute the nanmin (which is the minimum but ignores nan)
                nan_errors = np.where(errors == 0., np.nan, errors)
                new_minimum_signal_to_noise = np.nanmin(fluxes / nan_errors)
                new_maximum_signal_to_noise = np.nanmax(fluxes / nan_errors)
                new_minimum_flux = np.nanmin(fluxes)
                new_maximum_flux = np.nanmax(fluxes)

                # Only store these values if they're the most extreme we've found so far
                if new_minimum_signal_to_noise < self.photometry_stats.loc[a_flux_key, 'minimum_sn']:
                    self.photometry_stats.loc[a_flux_key, 'minimum_sn'] = new_minimum_signal_to_noise

                if new_maximum_signal_to_noise < self.photometry_stats.loc[a_flux_key, 'maximum_sn']:
                    self.photometry_stats.loc[a_flux_key, 'maximum_sn'] = new_maximum_signal_to_noise

                # Likewise for fluxes
                if new_minimum_flux < self.photometry_stats.loc[a_flux_key, 'minimum_flux']:
                    self.photometry_stats.loc[a_flux_key, 'minimum_flux'] = new_minimum_flux

                if new_maximum_flux < self.photometry_stats.loc[a_flux_key, 'maximum_flux']:
                    self.photometry_stats.loc[a_flux_key, 'maximum_flux'] = new_maximum_flux

                # Record mean errors by multiplying the current mean error by how many objects that came from, plus the
                # mean error of the current data multiplied by how many objects there are, all divided by the total
                # number of objects. I guess you could say it's a mean mean. ;)
                if first_run:
                    dict_of_all_errors[a_flux_key] = errors
                else:
                    dict_of_all_errors[a_flux_key] = np.append(dict_of_all_errors[a_flux_key], errors)

            # Reset first run tag so we now append errors to dicts instead
            first_run = False

        # Calculate medians  # todo: rename all this shit to median not mean lol
        for a_flux_key in flux_keys:
            self.photometry_stats.loc[a_flux_key, 'mean_error'] = np.nanmedian(dict_of_all_errors[a_flux_key])

        # We don't actually care about transforming if the smallest flux is greater than zero. In those cases, we reset
        # it to zero. Otherwise, we subtract a tiny extra amount so that we can never try to take a log(0).
        self.photometry_stats[['minimum_flux', 'minimum_sn']] = \
            np.where(self.photometry_stats[['minimum_flux', 'minimum_sn']] > 0, 0.,
                     self.photometry_stats[['minimum_flux', 'minimum_sn']] - 0.0001)

    @staticmethod
    def train_validation_split(data, training_set_size: float=0.75, seed: Optional[int]=42):
        """Randomly shuffles and splits the dataset into a training set and a validation set. Preferable for use over
        the sklearn module as it keeps the pandas data frame alive (including all our other information.)

        Args:/
            data (pd.DataFrame): dataset to split.
            training_set_size (float): size of training set to use. Default is 0.75.
            seed (float, int or None): random seed to use when splitting. Default is 42. None will set no seed.

        Returns:
            [a training data set, a validation data set]
        """
        # Set the numpy seed
        if seed is not None:
            np.random.seed(seed)

        # Make and shuffle and array of all indices of rows in the data frame
        data_indices = np.asarray(data.index)
        np.random.shuffle(data_indices)

        # Round to calculate the size of the training set we need
        n_objects = data.shape[0]
        training_data_size = int(np.around(n_objects * training_set_size))

        # Split and return the data set using the shuffled indices
        training_data = data.iloc[data_indices[:training_data_size]].reset_index(drop=True)
        validation_data = data.iloc[data_indices[training_data_size:]].reset_index(drop=True)
        return [training_data, validation_data]

    def enlarge_dataset_within_error(self, data, flux_keys: list, error_keys: list,
                                     min_sn: float = 1.0, max_sn: float = 2.0,
                                     error_model: str = 'exponential',
                                     error_correlation: Optional[str] = 'row-wise',
                                     outlier_model: Optional[str] = None,
                                     new_dataset_size_factor: Optional[float] = 10.,
                                     dataset_scaling_method: Optional[str]='random',
                                     edsd_mean_redshift: float=1.5,
                                     clip_fluxes: bool=True,
                                     clip_errors: bool=True,
                                     seed: Optional[int] = 42):
        """Function to create new data perturbed from original data. Can be very useful for artificially adding higher
        S/N data to the training data set.

        Dataset scaling methods - 'random', 'EDSD' or None

        Implemented error models - 'exponential', 'uniform'

        Implemented error correlations - 'row-wise' or None

        Implemented outlier models - None

        Args:
            data (pd.DataFrame): data to act on.
            flux_keys (list of str): names of fluxes in said data frame.
            error_keys (list of str): names of errors in said data frame.
            min_sn (float): minimum signal to noise to apply. Default is 1.0.
            max_sn (float): maximum signal to noise to apply. Must be larger than min_sn. Default is 1.0.
            error_model (str): frequency of perturbations to apply to model. Default is exponential.
            error_correlation (str or None): type of correlation to apply to errors. Default is row-wise (so, each row
                gets multiplied by deviates of the same standard deviation.)
            outlier_model (None or str): type of model to use to introduce outliers. Default is none. #todo implement?
            new_dataset_size_factor (float): factor to increase size of dataset by. Default is 10.
            dataset_scaling_method (str): what to use to upscale the dataset. Default is random.
            edsd_mean_redshift (float): mean redshift to be used by the edsd dataset upscaler.
            clip_fluxes (bool): whether or not to ensure no fluxes are less than band minimums. Necessary for log
                methods. Default is True.
            clip_errors (bool): whether or not to ensure no fluxes are less than band minimums. Necessary for log
                methods. Default is True.
            seed (float, int or None): seed to use for random points. Default is 42.

        Returns:
            a modified DataFrame with new values added.
        """

        # Set the numpy seed
        if seed is not None:
            np.random.seed(seed)

        # Get some useful stats
        n_rows = data.shape[0]
        n_columns = len(error_keys)

        # DATASET SCALING
        # Firstly, we do none if either no method or no new size has been specified
        if (dataset_scaling_method is None) or (new_dataset_size_factor is None):
            new_dataset_size = n_rows
            rows_to_use = np.arange(0, n_rows)

        # Decide which rows to pick from by pulling out a list of random rows to use
        elif dataset_scaling_method == 'random':
            new_dataset_size = int(np.round(n_rows * new_dataset_size_factor))
            rows_to_use = np.random.randint(0, n_rows, new_dataset_size)

        # Decide which rows to pick from by Monte-Carloing to pick rows to use based on a z^2 * e^(-z / L) curve
        elif dataset_scaling_method == 'edsd' or dataset_scaling_method == 'EDSD':
            new_dataset_size = int(np.round(n_rows * new_dataset_size_factor))

            # Define the curve, where z is the redshift and l is the mode of the distribution
            def r2_exp(z, l):
                return z**2 * np.exp(-z / (0.5 * l))

            # Grab the max of the distribution
            shift = 0.5
            edsd_mean_redshift += shift
            edsd_max = r2_exp(edsd_mean_redshift, edsd_mean_redshift)
            z_max = np.max(data['z_spec'])
            z_min = np.min(data['z_spec'])
            z_range = z_max - z_min

            # Loop, pulling out random variates from the r2_exp distribution
            random_deviate_redshifts = np.zeros(new_dataset_size, dtype=np.float)
            i = 0
            while i < new_dataset_size:
                redshift_to_try = z_range * np.random.rand() + z_min
                probability = edsd_max * np.random.rand()

                if probability < r2_exp(redshift_to_try + shift, edsd_mean_redshift):
                    random_deviate_redshifts[i] = redshift_to_try
                    i += 1

            # Find the argument of the nearest true redshift to inferred ones
            z_spec_transposed = data['z_spec'].values.reshape(-1, 1)
            rows_to_use = np.argmin(np.abs(z_spec_transposed - random_deviate_redshifts), axis=0)

        else:
            raise NotImplementedError('selected dataset_scaling_method not found.')

        # Next, let's make a new dataframe to perturb from the original one
        expanded_data = data.loc[rows_to_use].copy()

        # Pull out some random Gaussian values to multiply our lovely values with - we clip negative std deviations to zero.
        gaussian_deviates = np.random.normal(loc=0.0, scale=np.clip(expanded_data[error_keys].values, 0., None),
                                             size=(new_dataset_size, n_columns))

        # Decide on the scheme of signal to noise perturbation to use
        if error_correlation == 'row-wise':
            signal_to_noise_shape = (new_dataset_size, 1)

        elif error_correlation is None:
            signal_to_noise_shape = (new_dataset_size, n_columns)

        else:
            raise ValueError('error correlation of type "{}" not implemented!'.format(error_correlation))

        # Create signal to noise deviates by drawing from different distributions:

        # EXPONENTIAL - we set the inverse rate parameter 1/lambda to half the difference between the minimum and
        # maximum sn, meaning that ~63% of points are within min_sn < x < max_sn. Exponential distributions inherently
        # always allow a tiny number of big points, this is the best compromise though (probably.) The mean signal to
        # noise deviate will be max_sn - min_sn.
        if error_model == 'exponential':
            signal_to_noise_multipliers = (np.random.exponential(scale=max_sn - min_sn, size=signal_to_noise_shape)
                                           + min_sn)

        # UNIFORM - does what it says on the tin. Uniform sn deviates in the range min_sn < x < max_sn.
        elif error_model == 'uniform':
            signal_to_noise_multipliers = (max_sn - min_sn) * np.random.random(size=signal_to_noise_shape) + min_sn

        else:
            raise ValueError('specified error model "{}" not implemented!')

        # And now to perturb the heck outta this guy, clipping it to not be less than the pre-existing minimum flux in
        # that column.
        if clip_fluxes:
            minimum_fluxes = self.photometry_stats['minimum_flux'].values
        else:
            minimum_fluxes = -np.inf
        expanded_data[flux_keys] = np.clip(expanded_data[flux_keys].values
                                           + gaussian_deviates * signal_to_noise_multipliers,
                                           a_min=minimum_fluxes, a_max=None)

        # If the minimum signal to noise is less than one, then we won't be idiots and imply that the underlying data
        # will actually genuinely have error that's less than the original quoted errors
        expanded_data[error_keys] = (expanded_data[error_keys].values * np.clip(signal_to_noise_multipliers,
                                                                                a_min=1., a_max=None))

        # Clip any errors that have signal to noise ratios that are too small, by calculating an error that doesn't
        # result in too small errors by using error = flux / s_n_ratio
        if clip_errors:
            expanded_data[error_keys] = np.where(
                expanded_data[flux_keys].values / expanded_data[error_keys].values
                < self.photometry_stats['minimum_sn'].values,
                expanded_data[flux_keys].values / self.photometry_stats['minimum_sn'].values + 0.0001,
                expanded_data[error_keys])

        # Record the extent to which we messed with things in a new column (we take the mean applied to each row so that
        # None type error correlation doesn't record a stupidly big new column.
        expanded_data['mean_sn_perturbation'] = np.mean(signal_to_noise_multipliers, axis=1)

        # Reset the index and return
        return expanded_data.reset_index(drop=True)

    def convert_to_log_sn_errors(self, data, flux_keys: list, error_keys: list):
        """Function for re-scaling errors to be in log signal to noise ratio space, which ought to be better to train on.

        Args:
            data (pd.DataFrame): dataframe to act on.
            flux_keys (list of str): list of flux keys, in order.
            error_keys (list of str): list of error keys, in the same order as flux_keys..

        Returns:
            The edited data frame :)
        """
        # Make sure we aren't being naughty and assigning things to the wrong data frame
        # data = data.copy()

        # Cycle over all the bands and convert them to log signal to noise
        for a_flux_key, a_error_key in zip(flux_keys, error_keys):

            # Check we haven't been passed any incorrect values (that would fuck everything right up lol)
            bad_fluxes = np.count_nonzero(np.where(data[a_flux_key] == -99., True, False))
            if bad_fluxes > 0:
                raise ValueError('some fluxes in band {} were invalid!'.format(a_flux_key))

            # We only divide when the errors aren't zero. Otherwise, we set it to the maximum signal to noise ratio for
            # this band.
            sig_noise = np.where(data[a_error_key] != 0.0, data[a_flux_key] / data[a_error_key], -1.)
            sig_noise = np.where(sig_noise == -1., np.max(sig_noise), sig_noise)

            # Take the log and save it, being careful to not take a log of a negative number if there are any.
            log_sig_noise = np.log(sig_noise - self.photometry_stats.loc[a_flux_key, 'minimum_sn'])

            # Set any pathetically small values from 0 signal to noise to a smol number (this shouldn't need to get
            # called but has on rare occasions been needed)
            log_sig_noise = np.where(log_sig_noise == -np.inf, -700., log_sig_noise)

            data[a_error_key] = log_sig_noise

        return data

    def convert_to_log_fluxes(self, data, flux_keys):
        """Converts fluxes to log fluxes (plus their minimum value to avoid nans.) Can help to make the flux distribution
        more Gaussian, which may help with training. Has checks to ensure it doesn't take log of 0 or less!

        Args:
            data (pd.DataFrame): dataframe to act on.
            flux_keys (list of str): list of flux keys, in order.

        Returns:
            The edited data frame :)
        """
        # Make sure we aren't being naughty and assigning things to the wrong data frame
        # data = data.copy()

        # Cycle over flux bands
        for a_flux_key in flux_keys:

            # Check we haven't been passed any incorrect values (that would fuck the log right up lol)
            bad_fluxes = np.count_nonzero(np.where(data[a_flux_key] == -99., True, False))
            if bad_fluxes > 0:
                raise ValueError('some fluxes in band {} were invalid!'.format(a_flux_key))

            # Set the minimum to 0.0001 and take a log
            log_fluxes = np.log(data[a_flux_key] - self.photometry_stats.loc[a_flux_key, 'minimum_flux'])

            # Set any pathetically small values from 0 flux to a smol number (this shouldn't need to get
            # called but has on rare occasions been needed)
            data[a_flux_key] = np.where(log_fluxes == -np.inf, -700., log_fluxes)

        return data

    def convert_to_asinh_magnitudes(self, data, flux_keys):
        """Converts fluxes to Lupton+1999 asinh magnitudes. Can help to make the flux distribution more Gaussian, which
        may help with training. The asinh magnitude can handle negative fluxes!

        Args:
            data (pd.DataFrame): dataframe to act on.
            flux_keys (list of str): list of flux keys, in order.

        Returns:
            The edited data frame :)
        """
        # Grab Pogson's ratio (about 1.08 but I'm being precise for fun)
        pogson_ratio = 2.5 * np.log10(np.e)

        # Cycle over flux bands
        for a_flux_key in flux_keys:

            # Check we haven't been passed any incorrect values (that would fuck the log right up lol)
            bad_fluxes = np.count_nonzero(np.where(data[a_flux_key] == -99., True, False))
            if bad_fluxes > 0:
                raise ValueError('some fluxes in band {} were invalid!'.format(a_flux_key))

            # Grab b (the softening parameter: the sqrt of the Pogson ratio multiplied by the mean 1 sigma error in this
            # band
            b = np.sqrt(pogson_ratio) * self.photometry_stats.loc[a_flux_key, 'mean_error']
            data[a_flux_key] = -1. * pogson_ratio * (np.arcsinh(data[a_flux_key] / (2*b)) + np.log(b))

        return data
