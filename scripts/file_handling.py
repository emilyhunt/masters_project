"""All functions used for handling files."""

import pandas as pd
from astropy.io import fits
from scipy.io import readsav as scipy_readsave
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
    data = scipy_readsave(target, python_dict=True, verbose=False)  # Set verbose=True for more info on the file

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


def read_fits(fits_file_location: str, columns_to_keep: list, fits_index_to_read: int=1, new_column_names: bool=None,
              get_flux_and_error_keys: bool=False):
    """Creates a new photometry table in the pandas framework given a fits file, a set of keys to keep and new names for
    them. By creating a new object, it's nicer and also not read-only.

    Args:
        fits_file_location (location of an astropy BinTableHDU object): table to read in.
        columns_to_keep (list of str): keys to bother keeping from the read table.
        fits_index_to_read (int): index of the fits file to read in. Default is 1 (0 is usually a header.)
        new_column_names (None or list of str): new key names to assign to elements.
        get_flux_and_error_keys (bool): whether or not to look for flux and error keys and return them.

    Returns:
        A list, containing:
            0. a scrumptuous and lovely pandas data frame.
            (1. a list of all the keys corresponding to flux (aka feed for a neural net.))
            (2. a list of all the keys corresponding to error (aka more feed for a neural net.))

    """
    # Step 0: read in the fits file
    fits_file = fits.open(fits_file_location)[fits_index_to_read]

    # Step 1: let's make a data frame
    data = pd.DataFrame()

    if new_column_names is None:
        new_column_names = columns_to_keep

    # We cycle over the fits file, also doing byte shit as they seem to be wrong otherwise
    for new_key, old_key in zip(new_column_names, columns_to_keep):
        data[new_key] = fits_file.data[old_key].byteswap().newbyteorder()

    # Step 2: make fresh lists of the flux and error keys, if requested
    if get_flux_and_error_keys:
        flux_list = []
        error_list = []

        for a_string in new_column_names:

            if 'f_' in a_string:
                flux_list.append(a_string)

            elif 'e_' in a_string and a_string != 'use_phot':
                error_list.append(a_string)

        return [data, flux_list, error_list]

    # Or, just return data
    else:
        return data