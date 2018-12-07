"""All functions for preprocessing input photometric data."""

import numpy as np
import sys
from typing import Optional, Union


def missing_data_handler(data, flux_keys: list, error_keys: list,
                         band_central_wavelengths: list, coverage_minimum: float=0.95,
                         valid_photometry_column: str='use_phot', z_spec_column: str='z_spec',
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
        missing_flux_handling (None or str): method to use to replace missing flux points.
        missing_error_handling (None or str): method to use to replace missing error points.

    Returns:
        A list containing data that has valid spectroscopic redshifts (0th element) and data without valid
        spectroscopic redshifts (1st element).
    """
    # Make sure we're using local copies of the flux and error keys
    flux_keys = flux_keys.copy()
    error_keys = error_keys.copy()

    # Have a look at how many good values are in all columns
    all_columns_to_check = flux_keys + error_keys
    column_coverage = np.zeros(len(all_columns_to_check), dtype=np.float)
    columns_to_remove = []

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
            columns_to_remove.append(a_flux_key)
            columns_to_remove.append(a_error_key)

        flux_keys.remove(a_flux_key)
        error_keys.remove(a_error_key)
        band_central_wavelengths.pop(band_i)

    print('{} out of {} columns do not have coverage over {}% on good sources.'
          .format(len(columns_to_remove), len(flux_keys), coverage_minimum * 100))
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
        # Set it to 1000 times the maximum error in the catalogue
        the_big_number = np.max(np.asarray(data[error_keys])) * 1000

        # Grab the fluxes that need fixing and make the fixeyness happen!
        errors_to_fix = np.where(data[error_keys] == -99.0, False, True)
        data[error_keys] = data[error_keys].where(errors_to_fix, other=the_big_number)
        print('{} of {} errors were modified to a value of {}.'
              .format(errors_to_fix.size - np.count_nonzero(errors_to_fix), errors_to_fix.size, the_big_number))

    else:
        raise ValueError('specified missing_error_handling not found/implemented/supported.')

    # Step 4: Split the data into sets with and without spectroscopic redshifts
    has_spec_z = np.where(np.asarray(data[z_spec_column]) != -99.0)[0]
    data_spec_z = data.iloc[has_spec_z].copy().reset_index(drop=True)
    data_no_spec_z = data.drop(index=has_spec_z).copy().reset_index(drop=True)
    print('{} out of {} galaxies have spectroscopic redshifts.'.format(has_spec_z.size, data.shape[0]))

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
        # Set values to inf so the first run will always store new ones
        self.minimum_signal_to_noise = {}
        self.minimum_flux = {}

        # We loop over all data frames in all_input_data to find the most extreme point
        for a_data_frame in all_input_data:
            for a_flux_key, a_error_key in zip(flux_keys, error_keys):
                # Firstly, let's record the absolute smallest signal to noise ratio (it may well be negative)
                fluxes = a_data_frame[a_flux_key].values
                errors = a_data_frame[a_error_key].values

                # Check we haven't been passed any incorrect values (that would fuck everything right up lol)
                bad_fluxes = np.count_nonzero(np.where(fluxes == -99., True, False))
                if bad_fluxes > 0:
                    raise ValueError('some fluxes were invalid!')

                # Set any errors that are zero to np.nan, then compute the nanmin (which is the minimum but ignores nan)
                errors = np.where(errors == 0., np.nan, errors)
                new_minimum_signal_to_noise = np.nanmin(fluxes / errors)
                new_minimum_flux = np.nanmin(fluxes)

                # Only store these values if they're the most extreme we've found so far
                if a_error_key not in self.minimum_signal_to_noise.keys():
                    self.minimum_signal_to_noise[a_error_key] = new_minimum_signal_to_noise

                elif new_minimum_signal_to_noise < self.minimum_signal_to_noise[a_error_key]:
                    self.minimum_signal_to_noise[a_error_key] = new_minimum_signal_to_noise

                # Likewise for fluxes
                if a_flux_key not in self.minimum_flux.keys():
                    self.minimum_flux[a_flux_key] = new_minimum_flux

                elif new_minimum_flux < self.minimum_flux[a_flux_key]:
                    self.minimum_flux[a_flux_key] = new_minimum_flux

        # We don't actually care about transforming if the smallest flux is greater than zero. In those cases, we reset
        # it to zero. Otherwise, we subtract a tiny extra amount so that we can never try to take a log(0).
        for a_key in self.minimum_flux.keys():
            if self.minimum_flux[a_key] > 0:
                self.minimum_flux[a_key] = 0
            else:
                self.minimum_flux[a_key] -= 0.0001

        for a_key in self.minimum_signal_to_noise.keys():
            if self.minimum_signal_to_noise[a_key] > 0:
                self.minimum_signal_to_noise[a_key] = 0
            else:
                self.minimum_signal_to_noise[a_key] -= 0.0001

    @staticmethod
    def train_validation_split(data, training_set_size: float=0.75, seed: Optional[Union[float, int]]=42):
        """Randomly shuffles and splits the dataset into a training set and a validation set. Preferable for use over
        the sklearn module as it keeps the pandas data frame alive (including all our other information.)

        Args:
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

    def enlarge_dataset_within_error(self, data, min_sn: float=1.0, max_sn: float=2.0):
        """Function to create new data perturbed from original data. Can be very useful for artificially adding higher
        S/N data to the training data set."""
        pass

        # todo finish your fucking code you are so lazy lol

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
            log_sig_noise = np.log(sig_noise - self.minimum_signal_to_noise[a_error_key])

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
            log_fluxes = np.log(data[a_flux_key] - self.minimum_flux[a_flux_key])

            # Set any pathetically small values from 0 flux to a smol number (this shouldn't need to get
            # called but has on rare occasions been needed)
            data[a_flux_key] = np.where(log_fluxes == -np.inf, -700., log_fluxes)

        return data
