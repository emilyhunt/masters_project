"""Various plotting tools for redshift analysis, to be used extensively with z_util.py."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
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


def single_gaussian_to_fit(x, standard_deviation, A):
    """Allows a Gaussian to be accessed to fit a curve to."""
    return A * norm.pdf(x, loc=0, scale=standard_deviation)


def double_gaussian_to_fit(x, standard_deviation_1, standard_deviation_2, A, r):
    """Allows a double Gaussian convolved with another Gaussian to be accessed to fit a curve to."""
    sqrt_2 = np.sqrt(2)
    term_1 = A * norm.pdf(x, loc=0, scale=sqrt_2*standard_deviation_1)
    term_2 = A * 2 * r * norm.pdf(x, loc=0, scale=np.sqrt(standard_deviation_1**2 + standard_deviation_2**2))
    term_3 = A * r**2 * norm.pdf(x, loc=0, scale=sqrt_2 * standard_deviation_2)
    return term_1 + term_2 + term_3


def fit_gaussians(x_range, y_range):
    """Function that handles fitting Gaussians to our final data."""
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


def pair_redshift_deviation(my_redshifts, my_all_galaxy_pairs, my_random_galaxy_pairs, size_of_random_catalogue=5.):
    """Compares the difference in redshift between sky pairs and plots."""
    # Cast my_redshifts as a numpy array and cast the random catalogue size as a float
    my_redshifts = np.asarray(my_redshifts)
    random_redshifts = np.repeat(my_redshifts, size_of_random_catalogue)

    # Find all the pairs that don't contain any invalid redshifts (aka -99)
    #all_valid = np.where(np.logical_and(my_redshifts[my_all_galaxy_pairs[:, 0]] > 0,
    #                                    my_redshifts[my_all_galaxy_pairs[:, 1]] > 0))[0]
    #random_valid = np.where(np.logical_and(random_redshifts[my_random_galaxy_pairs[:, 0]] > 0,
    #                                       random_redshifts[my_random_galaxy_pairs[:, 1]] > 0))[0]

    # Calculate redshift difference over (1 add the mean redshift) for that galaxy
    z1 = my_redshifts[my_all_galaxy_pairs[:, 0]]
    z2 = my_redshifts[my_all_galaxy_pairs[:, 1]]
    all_redshift_difference = (z1 - z2) / (1 + 0.5 * (z1 + z2))
    print(all_redshift_difference.size)

    # Again for the random data set
    z1 = random_redshifts[my_random_galaxy_pairs[:, 0]]
    z2 = random_redshifts[my_random_galaxy_pairs[:, 1]]
    random_redshift_difference = (z1 - z2) / (1 + 0.5 * (z1 + z2))
    print(random_redshift_difference.size)

    # Histogram the different redshift pairs. density=True on the random pairs bins sets the area under the histogram to
    # unity, so that we can later multiply it back up again with a normalisation factor.
    all_redshifts_binned, bin_edges = np.histogram(all_redshift_difference, bins='auto')
    random_redshifts_binned, bin_edges = np.histogram(random_redshift_difference, bins=bin_edges, density=False)

    # Normalise the random redshifts by the factor of how many more there are vs real ones
    #n_data_points = my_redshifts.size
    #n_random_pairs_in_redshift_range = random_redshift_difference.size
    #np.sort(my_random_galaxy_pairs, axis=1)  # Sort all galaxy pairs into lowest ID, highest ID
    #n_unique_random_pairs = np.unique(np.sort(my_random_galaxy_pairs, axis=1), axis=0).size / 2
    #n_unique_all_pairs = np.unique(np.sort(my_all_galaxy_pairs, axis=1), axis=0).size / 2
    random_redshifts_binned = random_redshifts_binned * all_redshift_difference.size / random_redshift_difference.size

    # Subtract one from the other to get an estimate of the physical pair number
    physical_redshifts_binned = all_redshifts_binned - random_redshifts_binned

    # Work out the centre of the bins: by adding half of the bin spacing to all but the last bin edge
    bin_centres = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

    # Fit Gaussians to the data
    params = fit_gaussians(bin_centres, physical_redshifts_binned)

    # Calculate values of the Gaussians at different points
    fit_x_range = np.linspace(bin_edges[0], bin_edges[-1], 100)
    fit_y_range_1 = single_gaussian_to_fit(fit_x_range, params['s_s'], params['s_A'])
    fit_y_range_2 = double_gaussian_to_fit(fit_x_range, params['d_s1'], params['d_s2'], params['d_A'], params['d_r'])

    # Define our figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot stuff on the left
    ax_left.plot(bin_centres, all_redshifts_binned, 'r--', ms=3, label='All pairs')
    ax_left.plot(bin_centres, random_redshifts_binned, 'b--', ms=3, label='Random pairs')

    # Plot stuff on the right
    ax_right.plot(bin_centres, physical_redshifts_binned, 'k-', ms=3, label='Real distribution')
    ax_right.plot(fit_x_range, fit_y_range_1, 'r--', ms=3, label='Single Gaussian fit')
    ax_right.plot(fit_x_range, fit_y_range_2, 'b--', ms=3, label='Double Gaussian fit')

    # Pop some data on the plot
    #plt.text(np.min(bin_edges), np.max(physical_redshifts_binned),
    #         r'$mean(\Delta z / 1 + z_{mean})$ = '
    #         + r'{:.4f}'.format(mean_difference)
    #         + '\n' + r'$\sigma_{ \Delta z}$ = '
    #         + r'{:.4f}'.format(std_dev_difference),
    #         ha='left', va='top',
    #         bbox=dict(boxstyle='round', ec='k', fc='w'))

    # Prettify & show the plot
    ax_left.set_xlabel(r'$\Delta z / 1 + z_{mean}$')
    ax_right.set_xlabel(r'$\Delta z / 1 + z_{mean}$')
    ax_left.set_ylabel(r'$N_{pairs}$')
    ax_left.set_xlim(bin_edges[0], bin_edges[-1])
    ax_right.set_xlim(bin_edges[0], bin_edges[-1])
    ax_left.legend(edgecolor='k', facecolor='w', fancybox=True)
    ax_right.legend(edgecolor='k', facecolor='w', fancybox=True)
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

    # Remove everything within specific sky co-ordinates
    # redshifts = redshifts.iloc[redshifts.gs4_ra.values < 53.1]

    # Emily's diagnostic plot
    ra_res = dec_res = 30
    ra_range = np.linspace(redshifts.gs4_ra.values.min(), redshifts.gs4_ra.values.max(), ra_res)
    dec_range = np.linspace(redshifts.gs4_dec.values.min(), redshifts.gs4_dec.values.max(), dec_res)
    ra_space = np.abs(ra_range[1] - ra_range[0])
    dec_space = np.abs(dec_range[1] - dec_range[0])

    ra_grid, dec_grid = np.meshgrid(ra_range, dec_range)

    mean_redshifts = np.zeros(ra_grid.shape)

    i=0
    for ra in ra_range:
        j=0
        for dec in dec_range:

            good_ra = np.logical_and(redshifts.gs4_ra.values < ra + ra_space, redshifts.gs4_ra.values > ra)
            good_dec = np.logical_and(redshifts.gs4_dec.values < dec + dec_space, redshifts.gs4_dec.values > dec)
            good_zs = redshifts.gs4_zphot.values > 0
            good_coords = np.logical_and(good_ra, good_dec)
            good_final = np.logical_and(good_coords, good_zs)

            mean_redshifts[i, j] = np.mean(redshifts.gs4_zphot.values[good_final])
            j += 1
        i += 1

    plt.contourf(ra_grid, dec_grid, mean_redshifts.T, 50)
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.colorbar()
    plt.show()


    # Make a plot of locations on the sky
    sky_locations(redshifts['gs4_ra'], redshifts['gs4_dec'])

    """
    # Find all galaxy pairs
    random_catalogue_repeats = 1
    all_galaxy_pairs, random_galaxy_pairs = z_util.store_pairs_on_sky(redshifts['gs4_ra'][:],
                                                                      redshifts['gs4_dec'][:],
                                                                      max_separation=5., min_separation=0.0,
                                                                      size_of_random_catalogue=random_catalogue_repeats)

    # Try reading in the pairs again to check the storing worked
    max_z = 100
    min_z = 0
    all_galaxy_pairs_read_in = z_util.read_pairs('../data/all_pairs.csv', redshifts['gs4_zphot'],
                                                 min_redshift=min_z, max_redshift=max_z)

    random_galaxy_pairs_read_in = z_util.read_pairs('../data/random_pairs.csv', redshifts['gs4_zphot'],
                                                    min_redshift=min_z, max_redshift=max_z)

    # Make a plot of Npairs against deltaZ
    pair_redshift_deviation(redshifts['gs4_zphot'], all_galaxy_pairs_read_in, random_galaxy_pairs_read_in,
                            size_of_random_catalogue=random_catalogue_repeats)
    """