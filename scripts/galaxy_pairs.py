import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from typing import Union


def store_pairs_on_sky(ra, dec, max_separation: float=15.0, min_separation: float=0.0,
                       save_location: str='./data/', all_pairs_name: str='all_pairs.csv',
                       random_pairs_name: str='random_pairs.csv', random_seed: Union[int, float]=42,
                       size_of_random_catalogue: float=1., min_move: float=30., max_move: float=45.) -> list:
    """Wrapper for find_pairs_on_sky that finds physical x projected and just projected pairs, and then writes
    its findings to a file for later use.

    Args:
        ra (list-like): list of right ascensions of galaxies.
        dec (list-like): list of declinations of galaxies.
        max_separation (float): maximum separation of pairs to consider in arcseconds. Default is 15.0.
        min_separation (float): minimum separation of pairs to consider in arcseconds, sometimes useful when dealing
            with e.g. photometric contamination. Default is 0.0.
        save_location (str): folder where files should be saved. Default is './data/'.
        all_pairs_name (str): name of .csv file containing all pairs found. Default is 'all_pairs.csv'.
        random_pairs_name (str): name of .csv file containing all random found pairs. Default is 'random_pairs.csv'.
        random_seed (int, float): seed to use for random catalogue generation. Default is 42.
        size_of_random_catalogue (float): multiplier to use to make the random catalogue larger in number than the
            original one. Useful if too few galaxies are in the data set to make a good random cat from. Default is 1.0.
        min_move (float): minimum amount to move a galaxy by when creating the random catalogue. Default is 30.0.
        max_move (float): maximum amount to move a galaxy by when creating the random catalogue. Default is 30.0.

    Returns:
        a list of contents [all_pairs, random_pairs]

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
    np.repeat(catalogue, size_of_random_catalogue, axis=1)
    np.random.shuffle(catalogue.T)

    # We make a set of random numbers, between the minimum and maximum moves, and then multiply them by +1 or -1
    random_numbers = (max_move - min_move) * np.random.rand(2, catalogue[0].size) + min_move
    signs = np.random.rand(2, catalogue[0].size) - 0.5
    random_numbers = random_numbers * np.where(signs > 0, 1, -1) / 60**2  # The 60^2 converts from arcsec to deg

    # Now multiply the catalogue by said deviates, moving everything by between a minimum and maximum amount
    catalogue = catalogue + random_numbers

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
    """Identifies potential galaxy pairs on the sky and saves them to a file, using
    astropy.coordinates.search_around_sky.

    Args:
        catalogue (list-like): a list of all right ascensions and declinations, of form [ra_list, dec_list].
        max_separation (float): maximum separation of pairs to consider in arcseconds. Default is 15.0.
        min_separation (float): minimum separation of pairs to consider in arcseconds, sometimes useful when dealing
            with e.g. photometric contamination. Default is 0.0.

    Returns:
        a 2 x number_of_pairs list of all unique pairs found.
    """
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


def read_pairs(pair_location, redshifts, min_redshift=0, max_redshift=100, absolute_min_redshift=0, input_ids=None,
               size_of_random_catalogue=1):
    """Uses existing pair data files to quickly find pairs within specified categories. Takes pair_location as a string
    pointing to a file, or as an array of galaxy pair IDs. It will search for instances of 'IDs' in the list of pairs,
    returning the ID and the galaxy it is paired in a long array of valid pairs. It also supports restricting the
    redshift range. If reading IDs in as an array, it should have shape 2 x N_galaxies.
    """
    # Typecast redshifts as a numpy array
    redshifts = np.repeat(np.array(redshifts), size_of_random_catalogue)

    # If the user has specified a file with a string, then we want to read that in
    if isinstance(pair_location, str):
        matches = pd.read_csv(pair_location).values
        matches1 = matches[:, 0]
        matches2 = matches[:, 1]

    # Otherwise, pair_location should be a 2 x N_galaxies array that we get both rows of matches from
    else:
        matches1 = pair_location[0]
        matches2 = pair_location[1]

    # If the user has given a specific list of ids then we'll want to use those - otherwise, just create a single long
    # list of ids
    if input_ids is None:
        input_ids = np.arange(0, redshifts.size)

    # Find all galaxies that are desired and in the correct redshift range
    correct_z_input_ids = np.where(np.logical_and(redshifts[input_ids] < max_redshift,
                                                  redshifts[input_ids] > min_redshift))[0]

    # Find all instance of input IDs in the first and second columns of the pairs table (aka matches1 and matches2)
    id_1 = np.where(np.isin(matches1, correct_z_input_ids))[0]
    id_2 = np.where(np.isin(matches2, correct_z_input_ids))[0]

    # Create a new shortened list of pairs that satisfy our input conditions.  We'll need to not only use ID matches to
    # grab our galaxies, but also grab the IDs of the galaxies they're paired with.  We'll keep every ID requested
    # galaxy in the first column, and every galaxy it matches with in the second.
    paired_galaxies = np.empty((2, id_1.size + id_2.size), dtype=np.int64)
    paired_galaxies[0] = np.concatenate((matches1[id_1], matches2[id_2]))
    paired_galaxies[1] = np.concatenate((matches2[id_1], matches1[id_2]))

    # Find all pair galaxies that are not below the absolute minimum redshift (aka they're -99 or similar)
    incorrect_z_pair_ids = np.where(redshifts[paired_galaxies[1]] <= absolute_min_redshift)[0]
    paired_galaxies = np.delete(paired_galaxies, incorrect_z_pair_ids, axis=1)

    return paired_galaxies.T
