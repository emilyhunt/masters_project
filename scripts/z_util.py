"""A set of useful utilities for redshift calculations."""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
from astropy import units as u
from scipy.io.idl import readsav as scipy_readsav


def read_save(target, columns_to_keep=None, new_column_names=None):
    """Reads in a save file sensibly for use.
    columns_to_keep should be a list of columns, or the string 'only new ones' if the fn should... only keep new ones.
    new_column_names should be a dictionary with old names as keys and new names as values.
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


def calculate_nmad(spectroscopic_z, photometric_z):
    """Returns the normalised median absolute deviation between photometric and spectroscopic redshifts"""
    # Work out where there are valid spec and phot values
    valid = np.where(np.logical_and(spectroscopic_z > 0, photometric_z > 0))[0]

    # Calculate & return the NMAD
    return 1.4826 * np.median(np.abs((photometric_z[valid] - spectroscopic_z[valid]) / (1 + photometric_z[valid])))


def store_pairs_on_sky(ra, dec, max_separation=15.0, min_separation=0.0,
                       save_location='../data/', random_seed=42, all_pairs_name='all_pairs.csv',
                       random_pairs_name='random_pairs.csv', size_of_random_catalogue=5):
    """Wrapper for find_pairs_on_sky that finds physicalxprojected and just projected pairs, and then writes its
    findings to a file.
    """
    # Some shit about letting the user know what the code is doing
    print('Finding galaxy pairs within {} to {} arcseconds of each other.'.format(min_separation, max_separation))
    print('There are {} galaxies to test.'.format(ra.size))
    print('Please bear with, this could take some time!')

    # Typecast the ra/dec catalogue objects as np arrays
    catalogue = np.asarray([ra, dec])

    # First, let's find the union of physical and projected pairs
    all_pairs = find_pairs_on_sky(catalogue, max_separation, min_separation)

    # Next, let's create a random catalogue of co-ordinates so we can make a set of projected pairs
    # First, some housekeeping
    np.random.seed(random_seed)

    # Make a random catalogue by simply shuffling existing values. This ensures the random catalogue has an identical
    # distribution of ras and decs, but all at random redshifts to isolate projected pairs.
    np.random.shuffle(catalogue.T)

    # Run find_pairs_on_sky now with the random fun times
    random_pairs = find_pairs_on_sky(catalogue, max_separation, min_separation)

    # Make the number of randomised pairs size_of_random_catalogue times larger, and shuffle the order of pairs too
    # (shuffling the pairs order does nothing here, but does mean randomised redshifts will be picked later)
    random_pairs = np.repeat(random_pairs, size_of_random_catalogue, axis=0)
    np.random.shuffle(random_pairs.T)

    # Time to store this stuff. We use pandas as it seems by far the fastest way.
    pd.DataFrame(all_pairs).to_csv(save_location + all_pairs_name, index=False)
    pd.DataFrame(random_pairs).to_csv(save_location + random_pairs_name, index=False)

    # Return the pairs just in case the user can't be bothered to re-load them lmao
    return [all_pairs, random_pairs]


def find_pairs_on_sky(catalogue, max_separation=15.0, min_separation=0.0):
    """Identifies potential galaxy pairs on the sky and saves them to a file. Takes a while to run. Good luck!"""
    # Typecast the separation into astropy units
    max_separation = max_separation * u.arcsec
    min_separation = min_separation * u.arcsec

    # Set up our co-ordinates as an astropy SkyCoord object
    catalogue = SkyCoord(catalogue[0], catalogue[1], unit='deg')

    # Find galaxies paired with each other within separation
    matches1, matches2, angular_separation, physical_separation = search_around_sky(catalogue, catalogue,
                                                                                    seplimit=max_separation)

    # Remove matches where the angular separation is lower than the minimum separation
    small_matches = np.where(angular_separation < min_separation)[0]
    matches1 = np.delete(matches1, small_matches)
    matches2 = np.delete(matches2, small_matches)

    # Remove duplicates where matches1 = matches2
    matches_with_self = np.where(matches1 == matches2)[0]
    matches1 = np.delete(matches1, matches_with_self)
    matches2 = np.delete(matches2, matches_with_self)

    # Make it into one big array
    paired_galaxies = np.asarray([matches1, matches2]).T

    # Sort the array so pairs are lower ID - higher ID
    paired_galaxies = np.sort(paired_galaxies, axis=1)

    # Now that the IDs are sorted, making this a lot easier, remove any non-unique pairs
    paired_galaxies = np.unique(paired_galaxies, axis=0)

    return paired_galaxies


def read_pairs(input_ids, pair_location):
    """Uses existing pair data files to quickly find pairs within specified categories. Takes pair_location as a string
    pointing to a file, or as an array of galaxy pair IDs. It will search for instances of 'IDs' in the list of pairs.
    If reading in as an array, it should have shape 2 x N_galaxies.
    """
    # If the user has specified a file with a string, then we want to read that in
    if isinstance(pair_location, str):
        matches = pd.read_csv(pair_location).values
        matches1 = matches[:, 0]
        matches2 = matches[:, 1]

    # Otherwise, pair_location should be a 2 x N_galaxies array that we get both rows of matches from
    else:
        matches1 = pair_location[0]
        matches2 = pair_location[1]

    # Find all instance of IDs in the matches arrays
    id_1 = np.where(np.isin(matches1, input_ids))[0]
    id_2 = np.where(np.isin(matches2, input_ids))[0]

    # Create a new shortened list of pairs that satisfy our input conditions.  We'll need to not only use ID matches to
    # grab our galaxies, but also grab the IDs of the galaxies they're paired with.  We'll keep every ID requested
    # galaxy in the first column, and every galaxy it matches with in the second.
    paired_galaxies = np.empty((2, id_1.size + id_2.size), dtype=np.int64)
    paired_galaxies[0] = np.concatenate((matches1[id_1], matches2[id_2]))
    paired_galaxies[1] = np.concatenate((matches2[id_1], matches1[id_2]))

    return paired_galaxies.T
