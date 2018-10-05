import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
from astropy import units as u
from scipy.io.idl import readsav as scipy_readsav
from matplotlib import rc

# Make TeX labels work on plots
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def read_save(target, columns_to_keep=None):
    """Reads in a save file sensibly for use"""
    # Read in the specified save file
    data = scipy_readsav(target, python_dict=True, verbose=False)  # Set verbose=True for more info on the file

    # Remove the 'readme' key from the dictionary, if it exists - else return None
    data.pop('readme', None)

    # Ensure the byte order of the read in numpy arrays is the same as on this machine. Solves an issue as described at:
    # https://pandas.pydata.org/pandas-docs/stable/gotchas.html#byte-ordering-issues
    for a_key in data.keys():
        data[a_key] = data[a_key].byteswap().newbyteorder('L')

    # Collate the columns that we want and remove columns if the user has done anything with columns_to_keep
    if type(columns_to_keep) != type(None):
        return pd.DataFrame(data)[columns_to_keep]

    # If they haven't specified any cols to keep, then we just give them all of them
    else:
        return pd.DataFrame(data)


def plot_phot_vs_spec(spectroscopic_z, photometric_z, fig_name='phot_vs_spec.png', NMAD=None):
    """Plots photometric redshift against spectroscopic for analysis."""
    # Plot data points and a y=x bisector
    plt.figure()
    plt.plot(spectroscopic_z, photometric_z, 'r.', ms=2, alpha=0.2)
    plt.plot([-1, 10], [-1, 10], 'k--', lw=1)

    # Add the NMAD to the plot if it has been specified by the user
    if type(NMAD) != type(None):
        plt.text(1, 6.5, 'NMAD = {:.4f}'.format(NMAD), ha='center', va='center',
                 bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),))

    # Make it pwetty
    plt.xlim([-0.5, 7])
    plt.ylim([-0.5, 7])
    plt.xlabel(r'$z_{spec}$')
    plt.ylabel(r'$z_{phot}$')

    # Output time
    plt.savefig(fig_name)
    plt.show()
    return 0


def calc_normalised_median_abs_deviation(spectroscopic_z, photometric_z):
    """Returns the normalised median absolute deviation between photometric and spectroscopic redshifts"""
    # Work out where there are valid spec and phot values
    valid = np.where(np.logical_and(spectroscopic_z > 0, photometric_z > 0))[0]

    # Calculate & return the NMAD
    return 1.4826 * np.median(np.abs((photometric_z[valid] - spectroscopic_z[valid]) / (1 + photometric_z[valid])))


def find_pairs_on_sky(ra, dec, separation=1.0, save_location='/../output/', output_name='identified_pairs.dat'):
    """Identifies potential galaxy pairs on the sky and saves them to a file. Takes a while to run. Good luck!"""
    # Warn the user this might take a while
    print('Finding galaxy pairs within {} arcseconds of each other.'.format(separation))
    print('There are {} galaxies to test.'.format(ra.size))
    print('Please bear with, this could take some time!')

    # Typecast the separation into astropy units
    separation = separation * u.arcsec

    # Set up our co-ordinates as an astropy SkyCoord object and initialise a numpy integer array of galaxy pairs
    coords = SkyCoord(ra, dec, unit='deg')

    # Find galaxies paired with each other within separation
    matches1, matches2, sep2d, dist3d = search_around_sky(coords, coords, seplimit=separation)

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

    # Tell the user all about it and return
    print('All done! I found {} pairs.'.format(paired_galaxies.shape[0]))
    print('NOT IMPLEMENTED: writing this to a file')
    return paired_galaxies


def plot_pair_redshift_deviation(my_redshifts, my_galaxy_pairs):
    """Compares the difference in redshift between sky pairs and plots."""
    # Cast my_redshifts as a numpy array
    my_redshifts = np.asarray(my_redshifts)

    # Find all the pairs that don't contain any invalid redshifts (aka -99)
    valid = np.where(np.logical_and(my_redshifts[my_galaxy_pairs[:, 0]] >= 0,
                                    my_redshifts[my_galaxy_pairs[:, 1]] >= 0))[0]

    # Calculate redshift difference
    redshift_difference = my_redshifts[my_galaxy_pairs[valid, 0]] - my_redshifts[my_galaxy_pairs[valid, 1]]

    # Do a bit of maths on the mean bin value and standard deviation
    mean_difference = np.mean(redshift_difference)
    std_dev_difference = np.std(redshift_difference)

    # Make a histogram plot
    plt.figure()
    plt.hist(redshift_difference, bins='auto', color='r')

    # Pop some data on the plot
    plt.text(2, 1000, r'$\bar{ \Delta z}$ = '
                      + r'{:.4f}'.format(mean_difference)
                      + '\n' + r'$\sigma_{ \Delta z}$ = '
                      + r'{:.4f}'.format(std_dev_difference),
             ha='center', va='center',
             bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0)))

    # Prettify & show the plot
    plt.xlabel(r'$\Delta z$')
    plt.ylabel(r'$N_{pairs}$')
    plt.show()
    return 0


"""A set of unit tests and stuff to let us run this script independently for plots if desired."""
if __name__ == '__main__':
    print('Script is being ran independently! Beginning unit tests:')

    # Variables for unit test
    data_dir = '/home/emily/uni_files/catalogs/KMOS415_output/'  # Reminder: put a slash on the end
    files_to_read = ['GS415.3dhst.redshift.save', 'GS415.coo.save']

    # Read in the .save files
    print('Reading in default save files')
    redshifts = read_save(data_dir + files_to_read[0])
    coords = read_save(data_dir + files_to_read[1])

    # Join the co-ordinates to the redshifts DataFrame
    redshifts['gs4_ra'] = coords['gs4_ra']
    redshifts['gs4_dec'] = coords['gs4_dec']

    # Calculate the NMAD
    my_NMAD = calc_normalised_median_abs_deviation(redshifts['gs4_zspec'], redshifts['gs4_zphot'])

    # Make a plot of the photometric redshifts against spectroscopic
    plot_phot_vs_spec(redshifts['gs4_zspec'], redshifts['gs4_zphot'], NMAD=my_NMAD)

    # Find all galaxy pairs
    galaxy_pairs = find_pairs_on_sky(redshifts['gs4_ra'], redshifts['gs4_dec'], separation=2.0)

    # Make a plot of Npairs against deltaZ
    plot_pair_redshift_deviation(redshifts['gs4_zphot'], galaxy_pairs)
