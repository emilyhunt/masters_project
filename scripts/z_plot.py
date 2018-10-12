"""Various plotting tools for redshift analysis, to be used extensively with z_util.py."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import rc
from scripts import z_util

# Make TeX labels work on plots
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def phot_vs_spec(spectroscopic_z, photometric_z, fig_name='phot_vs_spec.png', nmad=None):
    """Plots photometric redshift against spectroscopic for analysis."""
    # Plot data points and a y=x bisector
    plt.figure()
    plt.plot(spectroscopic_z, photometric_z, 'r.', ms=2, alpha=0.2)
    plt.plot([-1, 10], [-1, 10], 'k--', lw=1)

    # Add the NMAD to the plot if it has been specified by the user
    if nmad is not None:
        plt.text(1, 6.5, 'NMAD = {:.4f}'.format(nmad), ha='center', va='center',
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


def sky_locations(ra, dec):
    """Plots the sky locations of all galaxies requested, allowing for analysis of their place in space."""
    # Setup our projection
    plt.figure()

    # Data plotting and grid showing
    plt.plot(ra, dec, 'r.', ms=2, alpha=0.2)

    # Beautifying
    plt.xlabel('Right ascension')
    plt.ylabel('Declination')
    plt.show()
    return 0


def pair_redshift_deviation(my_redshifts, my_all_galaxy_pairs, my_random_galaxy_pairs):
    """Compares the difference in redshift between sky pairs and plots."""
    # Cast my_redshifts as a numpy array
    my_redshifts = np.asarray(my_redshifts)

    # Find all the pairs that don't contain any invalid redshifts (aka -99)
    all_valid = np.where(np.logical_and(my_redshifts[my_all_galaxy_pairs[:, 0]] > 0,
                                        my_redshifts[my_all_galaxy_pairs[:, 1]] > 0))[0]
    random_valid = np.where(np.logical_and(my_redshifts[my_random_galaxy_pairs[:, 0]] > 0,
                                           my_redshifts[my_random_galaxy_pairs[:, 1]] > 0))[0]

    # Calculate redshift difference over (1 add the mean redshift) for that galaxy
    z1 = my_redshifts[my_all_galaxy_pairs[all_valid, 0]]
    z2 = my_redshifts[my_all_galaxy_pairs[all_valid, 1]]
    all_redshift_difference = (z1 - z2) / (1 + 0.5*(z1 + z2))

    # Again for the random data set
    z1 = my_redshifts[my_random_galaxy_pairs[random_valid, 0]]
    z2 = my_redshifts[my_random_galaxy_pairs[random_valid, 1]]
    random_redshift_difference = (z1 - z2) / (1 + 0.5 * (z1 + z2))

    # Histogram the different redshift pairs
    all_redshifts_binned, bin_edges = np.histogram(all_redshift_difference, bins='auto')
    random_redshifts_binned, bin_edges = np.histogram(random_redshift_difference, bins=bin_edges)

    # Subtract one from the other to get an estimate of the physical pair number, also it's normalised
    physical_redshifts_binned = all_redshifts_binned - all_valid.size / random_valid.size * random_redshifts_binned

    # Work out the centre of the bins: by adding half of the bin spacing to all but the last bin edge
    bin_centres = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

    # Do a bit of maths on the mean bin value and standard deviation  todo: a Gaussian fit!
    mean_difference = -99
    std_dev_difference = -99

    # Make a histogram plot
    plt.figure()
    plt.plot(bin_centres, physical_redshifts_binned, 'k-', ms=3, label='Real distribution')
    plt.plot(bin_centres, all_redshifts_binned, 'r--', ms=3, label='All distribution')
    plt.plot(bin_centres, random_redshifts_binned, 'b--', ms=3, label='Random distribution')

    # Overplot a Gaussian
    #x_range = np.linspace(bin_edges[0] - 0.2, bin_edges[-1] + 0.2, num=100)  # Change num to change resolution
    #y_range = norm.pdf(x_range, loc=mean_difference, scale=std_dev_difference)
    #plt.plot(x_range, y_range, 'r-', label='Gaussian fit')

    # Pop some data on the plot
    #plt.text(np.min(bin_edges), np.max(physical_redshifts_binned),
    #         r'$mean(\Delta z / 1 + z_{mean})$ = '
    #         + r'{:.4f}'.format(mean_difference)
    #         + '\n' + r'$\sigma_{ \Delta z}$ = '
    #         + r'{:.4f}'.format(std_dev_difference),
    #         ha='left', va='top',
    #         bbox=dict(boxstyle='round', ec='k', fc='w'))

    # Prettify & show the plot
    plt.xlabel(r'$\Delta z / 1 + z_{mean}$')
    plt.ylabel(r'$N_{pairs}$')
    plt.xlim(bin_edges[0] - 0.2, bin_edges[-1] + 0.2)
    plt.legend(edgecolor='k', facecolor='w', fancybox=True)
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
    redshifts = z_util.read_save(data_dir + files_to_read[0])
    coords = z_util.read_save(data_dir + files_to_read[1])

    # Join the co-ordinates to the redshifts DataFrame
    redshifts['gs4_ra'] = coords['gs4_ra']
    redshifts['gs4_dec'] = coords['gs4_dec']

    # Calculate the NMAD
    my_nmad = z_util.calculate_nmad(redshifts['gs4_zspec'], redshifts['gs4_zphot'])

    # Make a plot of the photometric redshifts against spectroscopic
    #phot_vs_spec(redshifts['gs4_zspec'], redshifts['gs4_zphot'], nmad=my_nmad)

    # Make a plot of locations on the sky
    #sky_locations(redshifts['gs4_ra'], redshifts['gs4_dec'])

    # Cull redshifts if desired
    small_redshifts = np.where(np.logical_and(redshifts['gs4_zphot'] < 2, redshifts['gs4_zphot'] > 1.3))[0]

    # Find all galaxy pairs
    #all_galaxy_pairs, random_galaxy_pairs = z_util.store_pairs_on_sky([redshifts['gs4_ra'][small_redshifts],
    #                                          redshifts['gs4_dec'][small_redshifts]],
    #                                         max_separation=15., min_separation=3.)

    # Try reading in the pairs again to check the storing worked
    all_galaxy_pairs_read_in = z_util.read_pairs(small_redshifts, '../data/all_pairs.dat')
    random_galaxy_pairs_read_in = z_util.read_pairs(small_redshifts, '../data/random_pairs.dat')

    # Make a plot of Npairs against deltaZ
    pair_redshift_deviation(redshifts['gs4_zphot'], all_galaxy_pairs_read_in, random_galaxy_pairs_read_in)
