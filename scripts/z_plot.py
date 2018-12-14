"""Various plotting tools for redshift analysis, to be used extensively with util.py."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scripts.file_handling
import scripts.galaxy_pairs
from scripts import util
from typing import Optional, Union
from scipy.stats import norm as scipy_normal

# Make TeX labels work on plots
#rc('font', **{'family': 'serif', 'serif': ['Palatino']})
#rc('text', usetex=True)
from scripts.util import single_gaussian_to_fit, double_gaussian_to_fit, fit_gaussians


# Stop plots from popping up constantly when being updated (turn back on with plt.ion() for console plotting fun.)
plt.ioff()


def phot_vs_spec(spectroscopic_z, photometric_z, show_nmad: bool=True, nmad_bins: int=5, point_alpha: float=0.2,
                 point_color='r', limits=None,
                 plt_title: Optional[str]=None, save_name: Optional[str]=None, show_fig: bool=False,
                 validity_condition: str='greater_than_zero'):
    """Plots photometric redshift against spectroscopic for analysis. Also makes a plot of binned NMAD and median
    values, and makes a plot of zphot - zspec / (1 + zspec) for analysis.
    
    Args:
        Function-specific:
            spectroscopic_z (array-like): true redshifts to plot.
            photometric_z (array-like): inferred redshifts to plot.
            show_nmad (bool): whether or not to plot the nmad. Default is true.
            point_alpha (float): alpha of points to plot. Default is 0.2.
            point_color (matplotlib color): color of the points. Default is red ('r').
            limits (None or array-like): x and y limits of redshift range. Default is None.
        
        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.
            validity_condition (str): validity condition to pass to util.where_valid_redshifts().
            
    Returns:
        the best plot ever, and also the NMAD
            
    """  # todo this fn needs updating

    # Find valid redshifts and cast the arrays as only being said valid redshifts
    valid = util.where_valid_redshifts(spectroscopic_z, photometric_z, validity_condition=validity_condition)
    spectroscopic_z = np.asarray(spectroscopic_z)[valid]
    photometric_z = np.asarray(photometric_z)[valid]

    # Figure and axis setup
    fig = plt.figure(figsize=(7.48, 4))
    ax_spec_phot = fig.add_subplot(1, 2, 1)  # n_rows, n_cols, axis_number
    ax_nmad = fig.add_subplot(2, 2, 2)
    ax_delta_z = fig.add_subplot(2, 2, 4, sharex=ax_nmad)

    # LEFT PLOT - inferred/photometric redshifts vs true/spectroscopic redshifts.
    # Begin by plotting things:
    ax_spec_phot.plot(spectroscopic_z, photometric_z, '.', color=point_color, ms=2, alpha=point_alpha)
    ax_spec_phot.plot([-1, 10], [-1, 10], 'k--', lw=1)

    # Add the NMAD to the plot if it has been specified by the user
    if show_nmad is True:
        overall_nmad = util.calculate_nmad(spectroscopic_z, photometric_z)
        ax_spec_phot.text(0.05, 0.95, 'NMAD = {:.4f}'.format(overall_nmad),
                     ha='left', va='center', transform=ax_spec_phot.transAxes, fontsize=8,
                     bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),))
    else:
        overall_nmad = None

    # RIGHT PLOT 1 - NMAD vs redshift
    # Begin by sorting the redshifts into spectroscopic order
    sort_indices = np.argsort(spectroscopic_z)
    spec_sorted = spectroscopic_z[sort_indices]
    phot_sorted = photometric_z[sort_indices]

    # Now, split them into nmad_bins groups
    bin_size = spec_sorted.size // nmad_bins
    bin_mean_x_values = np.zeros(nmad_bins, dtype=np.float)
    bin_range = np.zeros((2, nmad_bins), dtype=np.float)
    binned_nmads = np.zeros(nmad_bins, dtype=np.float)
    binned_median_delta_z = np.zeros(nmad_bins, dtype=np.float)

    # Loop over each bin and calculate the NMAD for that range
    i = 0
    while i < nmad_bins:
        start_index = i * bin_size
        end_index = (i+1) * bin_size

        # Set end index to be the last value if we're on the last bin, to deal with remainders in the bin size integer
        # division
        if i == nmad_bins - 1:
            end_index = spec_sorted.size - 1

        # Calculate the NMAD here
        binned_nmads[i] = util.calculate_nmad(spec_sorted[start_index:end_index],
                                              phot_sorted[start_index:end_index])
        binned_median_delta_z[i] = np.median((phot_sorted[start_index:end_index] - spec_sorted[start_index:end_index])
                                      / (1 + spec_sorted[start_index:end_index]))

        # Store some extra stuff
        bin_mean_x_values[i] = np.mean(spec_sorted[start_index:end_index])
        bin_range[:, i] = np.abs(np.array([spec_sorted[start_index], spec_sorted[end_index]]) - bin_mean_x_values[i])

        i += 1

    ax_nmad.errorbar(bin_mean_x_values, binned_nmads, xerr=bin_range, fmt='ks', ms=3, label='NMAD')
    ax_nmad.errorbar(bin_mean_x_values, binned_median_delta_z, xerr=bin_range, fmt='bs', ms=3,
                     label='Median $\Delta z$')
    ax_nmad.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)

    # RIGHT PLOT 2 - Evolution of deltaZ / (1 + z) and binned NMADs
    delta_z = (photometric_z - spectroscopic_z) / (1 + spectroscopic_z)
    ax_delta_z.plot(spectroscopic_z, delta_z, '.', color=point_color, ms=2, alpha=point_alpha)

    # todo: make this prettier with a density plot. looks hard sadly. try using e.g.:
    # contour_plot = ax_delta_z.contourf(spectroscopic_z, delta_z, cmap='magma')
    # fig.colorbar(contour_plot)

    ax_delta_z.plot([-1, 10], [0, 0], 'k--', lw=1)

    # Set limits and labels
    if limits is not None:
        ax_spec_phot.set_xlim(limits[0], limits[1])
        ax_nmad.set_xlim(limits[0], limits[1])
        ax_spec_phot.set_ylim(limits[0], limits[1])
    ax_spec_phot.set_xlabel(r'$z_{spec}$')
    ax_spec_phot.set_ylabel(r'$z_{phot}$')

    ax_nmad.set_ylabel(r'NMAD, median $\Delta z / ( 1 + z_{spec})$')
    ax_delta_z.set_xlabel(r'$z_{spec}$')
    ax_delta_z.set_ylabel(r'$\Delta z / ( 1 + z_{spec})$')

    # Fix possible spacing issues between left and right subplots
    fig.subplots_adjust(wspace=0.3)

    # Output time
    if plt_title is not None:
        ax_spec_phot.set_title(plt_title)

    if save_name is not None:
        fig.savefig(save_name)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)

    return overall_nmad


def sky_locations(ra, dec):
    """Plots the sky locations of all galaxies requested, allowing for analysis of their place in space.

    Args:
        ra (array-like): right ascensions to plot.
        dec (array-like): declinations to plot.

    Returns:
        a plot!
    """
    # Setup our projection
    plt.figure()

    # Data plotting and grid showing
    plt.plot(ra, dec, 'r.', ms=2, alpha=0.2)

    # Beautifying
    plt.xlabel('Right ascension')
    plt.ylabel('Declination')
    plt.show()


def density_plot(x, y, z, n_levels: int=50, grid_resolution: int=30, x_lim: Optional[Union[tuple, list]]=None,
                 y_lim: Optional[Union[tuple, list]]=None, z_lim: Optional[Union[tuple, list]]=None,
                 plt_title: Optional[str] = None, save_name: Optional[str] = None, show_fig: bool = False,
                 x_label: Optional[str]='x', y_label: Optional[str]='y', point_size: float=1., point_alpha: float=0.05):
    """Makes a surface density plot of some particular desired parameter.

    Args:
        Function-specific:
            x (array-like): x values to plot.
            y (array-like): y values to plot.
            z (array-like): corresponding redshifts of objects.
            n_levels (int): number of levels to plot in the colour chart. Default is 50.
            grid_resolution (int): number of grid points to take in each direction. Default is 30.
            x_lim (None, list or tuple): allowed limits of x values.
            y_lim (None, list or tuple): allowed limits of x values.
            z_lim (None, list or tuple): allowed limits of x values.
            x_label (None or str): x label for the plot.
            y_label (None or str): y label for the plot.
        
        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.

    Returns:
        a plot!
    """
    # todo: needs updating and/or testing

    x_resolution = y_resolution = grid_resolution

    # Setup the limits, ignoring anything not within said limits
    if x_lim is None:
        allowed_x = np.ones(x.size, dtype=bool)
    else:
        allowed_x = np.where(np.logical_and(x_lim[0] < x, x < x_lim[1]), True, False)

    if y_lim is None:
        allowed_y = np.ones(y.size, dtype=bool)
    else:
        allowed_y = np.where(np.logical_and(y_lim[0] < y, y < y_lim[1]), True, False)

    if z_lim is None:
        allowed_z = np.ones(z.size, dtype=bool)
    else:
        allowed_z = np.where(np.logical_and(z_lim[0] < z, z < z_lim[1]), True, False)

    # Now, only keep values that satisfy all conditions
    allowed_values = np.logical_and(allowed_x, np.logical_and(allowed_y, allowed_z))
    x = x[allowed_values]
    y = y[allowed_values]
    z = z[allowed_values]

    # Calculate some ranges to grid over
    x_range = np.linspace(x.min(), x.max(), x_resolution)
    x_spacing = np.abs(x_range[1] - x_range[0])
    y_range = np.linspace(y.min(), y.max(), y_resolution)
    y_spacing = np.abs(y_range[1] - y_range[0])

    # Make grid points for later
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Cycle over the different grid points calculating the mean redshift in each place
    mean_z = np.zeros(x_grid.shape)
    i = 0
    for a_x in x_range:
        j = 0
        for a_y in y_range:
            good_x = np.logical_and(a_x < x, x < a_x + x_spacing)
            good_y = np.logical_and(a_y < y, y < a_y + y_spacing)
            good_z = z[np.logical_and(good_x, good_y)]

            # Only calculate the mean if there's more than one in this bin!
            if good_z.size > 0:
                mean_z[i, j] = np.mean(good_z)
            else:
                mean_z[i, j] = np.nan
            j += 1
        i += 1

    # Do the contoury bit
    fig = plt.figure(figsize=(7.48, 4))
    ax = fig.add_subplot(1, 2, 1)
    density_plot_object = ax.contourf(x_grid, y_grid, mean_z.T, n_levels, cmap=plt.get_cmap('plasma'))

    # Setup of the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(density_plot_object, cax=cax, orientation='vertical')

    # Make a separate plot showing the points going into the plot
    ax2 = fig.add_subplot(1, 2, 2, sharey=ax)
    ax2.plot(x, y, 'k.', ms=point_size, alpha=point_alpha)

    # Output time
    if x_label is not None:
        ax.set_xlabel(x_label)
        ax2.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    if plt_title is not None:
        ax.set_title(plt_title)
    if save_name is not None:
        fig.savefig(save_name)
    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def population_histogram(my_redshifts, x_label: str=r'$z_{phot}$', color: Optional[Union[str, float, tuple, list]]='r',
                         bins: Union[str, int]='auto',
                         plt_title: Optional[str]=None, save_name: Optional[str]=None, show_fig: bool=False,
                         validity_condition: str = 'greater_than_zero'):
    """Creates a histogram of the population of redshifts. Pretty basic stuff but hey, plotting functions are cool and
    I don't need your criticism here.

    Args:
        Function-specific:
            my_redshifts (array-like): redshifts to plot.
            color (array-like): the color of the histogram we're plotting.
            x_label (str): the x axis label to use. Default is z_phot.
            bins (int or str): the number of bins to use. Default is auto, which uses builtin optimum bin number methods
                in matplotlib/numpy.

        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.
            validity_condition (str): validity condition to pass to util.where_valid_redshifts().

    Returns:
        the best plot ever
    """
    # Find valid redshifts and cast the arrays as only being said valid redshifts
    valid = util.where_valid_redshifts(my_redshifts, validity_condition=validity_condition)
    my_redshifts = np.asarray(my_redshifts)[valid]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(my_redshifts, bins=bins, color=color)
    ax.set_ylabel('Frequency')

    # Output time
    if x_label is not None:
        ax.set_xlabel(x_label)

    if plt_title is not None:
        ax.set_title(plt_title)

    if save_name is not None:
        fig.savefig(save_name)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def pair_redshift_deviation(my_redshifts, my_all_galaxy_pairs, my_random_galaxy_pairs,
                            size_of_random_catalogue: float=1., plt_title: Optional[str]=None,
                            save_name: Optional[str]=None, show_fig: bool=False,
                            validity_condition: str='greater_than_zero'):
    """Compares the difference in redshift between sky pairs and plots the results. Makes a plot of both all pairs
    and the randomly distributed pairs, and then subtracts the two to get physical pairs only, and does some Gaussian
    fitting to this too.

    Args:
        Function-specific:
            my_redshifts (array-like): redshifts of all N sources.
            my_all_galaxy_pairs (array-like): all galaxy pairs, of shape Nx2.
            my_random_galaxy_pairs (array-like): randomly simulated galaxy pairs, of shape Nx2.
            size_of_random_catalogue (float): size of the random catalogue. Default is 1.0.

        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.
            validity_condition (str): validity condition to pass to util.where_valid_redshifts().

    Returns:
        a beautiful plot or your money back tbh

    """
    # Cast my_redshifts as a numpy array and make a random catalogue of the desired size
    valid = util.where_valid_redshifts(my_redshifts, my_redshifts, validity_condition=validity_condition)
    my_redshifts = np.asarray(my_redshifts)

    if valid.size != my_redshifts.size:
        print('WARNING: pair_redshift_deviation has been passed invalid redshifts! This is disallowed, as it does not'
              ' have the capability to remove shitty redshifts. Don\'t trust the next plot...')

    random_redshifts = np.repeat(my_redshifts, size_of_random_catalogue)

    # Calculate redshift difference over (1 add the mean redshift) for that galaxy
    z1 = my_redshifts[my_all_galaxy_pairs[:, 0]]
    z2 = my_redshifts[my_all_galaxy_pairs[:, 1]]
    all_redshift_difference = (z1 - z2) / (1 + 0.5 * (z1 + z2))

    # Again for the random data set
    z1 = random_redshifts[my_random_galaxy_pairs[:, 0]]
    z2 = random_redshifts[my_random_galaxy_pairs[:, 1]]
    random_redshift_difference = (z1 - z2) / (1 + 0.5 * (z1 + z2))

    # Histogram the different redshift pairs. density=True on the random pairs bins sets the area under the histogram to
    # unity, so that we can later multiply it back up again with a normalisation factor.
    all_redshifts_binned, bin_edges = np.histogram(all_redshift_difference, bins='auto')
    random_redshifts_binned, bin_edges = np.histogram(random_redshift_difference, bins=bin_edges, density=False)

    # Normalise the random redshifts by the factor of how many more there are vs real ones
    random_redshifts_binned = random_redshifts_binned / size_of_random_catalogue

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
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.48, 4))

    # Plot stuff on the left
    ax_left.plot(bin_centres, all_redshifts_binned, 'r--', ms=3, label='All pairs')
    ax_left.plot(bin_centres, random_redshifts_binned, 'b--', ms=3, label='Random pairs')

    # Plot stuff on the right
    ax_right.plot(bin_centres, physical_redshifts_binned, 'k-', ms=3, label='Phys. pairs')
    ax_right.plot(fit_x_range, fit_y_range_1, 'r--', ms=3, label='$\sigma = $ {:.2f}'.format(params['s_s']))
    ax_right.plot(fit_x_range, fit_y_range_2, 'b--', ms=3, label='$\sigma_1 = $ {:.2f}, $\sigma_2 = $ {:.2f}'
                                                                 .format(params['d_s1'], params['d_s2']))

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
    ax_left.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)
    ax_right.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)

    # Output time
    if plt_title is not None:
        ax_left.set_title(plt_title)

    if save_name is not None:
        fig.savefig(save_name)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def error_evaluator(spectroscopic_z, photometric_z, sigma_1_lower, sigma_1_upper=None, limits=(-5, 5),
                    plt_title: Optional[str]=None, save_name: Optional[str]=None, show_fig: bool=False,
                    validity_condition: str='greater_than_zero'):
    """Makes a histogram evaluating how good the given redshift error estimates are, and attempts to fit a Gaussian to
    this. A useful tool for evaluating the reliability of errors from a method.

    Args:
        Function-specific:
            spectroscopic_z (array-like): true redshifts to test against.
            photometric_z (array-like): inferred redshifts to test.
            sigma_1_lower (array-like): lower bounds on redshifts. Should be a positive
            sigma_1_upper (array-like): upper bounds on redshifts. Set to None, this will be a copy of sigma_1_low (useful
                for if error estimates are symmetrical.)
            limits (array-like): x limits for the plot.

        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.
            validity_condition (str): validity condition to pass to util.where_valid_redshifts().

    Returns:
        A gorgeous plot
    """
    # If no upper bounds are specified, set them to the lower bounds
    if sigma_1_upper is None:
        sigma_1_upper = sigma_1_lower

    # Find out which values are actually good, and cast everything as these
    valid = util.where_valid_redshifts(spectroscopic_z, photometric_z, validity_condition=validity_condition)
    spectroscopic_z = np.asarray(spectroscopic_z)[valid]
    photometric_z = np.asarray(photometric_z)[valid]
    sigma_1_lower = np.asarray(sigma_1_lower)[valid]
    sigma_1_upper = np.asarray(sigma_1_upper)[valid]

    # Calculate the difference between true and inferred values, and use this to decide which error to use using
    # a clever thing with signs and shit
    delta_z = photometric_z - spectroscopic_z  # +ve values => use upper, -ve values => use lower
    signs = np.sign(delta_z)
    use_lower = np.where(signs == -1)[0]
    use_upper = np.where(signs == +1)[0]
    deviation_in_sigmas = np.append(delta_z[use_lower] / (photometric_z[use_lower] - sigma_1_lower[use_lower]),
                                    delta_z[use_upper] / (sigma_1_upper[use_upper] - photometric_z[use_upper]))

    # Clip at +/- the limits
    if limits is not None:
        deviation_in_sigmas_clipped = deviation_in_sigmas[np.where(
            np.logical_and(deviation_in_sigmas >= limits[0], deviation_in_sigmas <= limits[1]))[0]]
    else:
        deviation_in_sigmas_clipped = deviation_in_sigmas

    # Bin the deviations into a histogram, using the automatically best number of bins, and get the bin centres
    deviation_in_sigmas_binned, bin_edges = np.histogram(deviation_in_sigmas_clipped, bins='auto', density=True)

    # Work out the centre of the bins: by adding half of the bin spacing to all but the last bin edge
    bin_centres = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

    # Fit a Normal to this
    params = util.fit_normal(bin_centres, deviation_in_sigmas_binned)

    # Calculate values of the Gaussians at different points
    fit_x_range = np.linspace(bin_edges[0], bin_edges[-1], 100)
    fit_y_range = util.single_normal_to_fit(fit_x_range, params['s_s'], params['s_m'])

    # Matplotlib shit
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(bin_centres, deviation_in_sigmas_binned, 'k-', label='Data')
    ax.plot(fit_x_range, fit_y_range, 'r--',
             label=r'Fit: $\mu=$' + '{:.2f}'.format(params['s_m']) + r', $\sigma=$' + '{:.2f}'.format(params['s_s']))
    ax.plot(fit_x_range, util.single_normal_to_fit(fit_x_range, 1, 0), 'b--',
             label=r'Fit: $\mu=0$, $\sigma=1$')

    # r'$\mu = $' + '{}'.format(params['s_m'])

    # Pop, in text, the fractions of everything at different sigma levels
    total = deviation_in_sigmas.size
    sigma_level_1 = np.count_nonzero(np.where(np.abs(deviation_in_sigmas) < 1., True, False)) / total * 100  # 68.3%
    sigma_level_2 = np.count_nonzero(np.where(np.abs(deviation_in_sigmas) < 2., True, False)) / total * 100  # 95.5%
    sigma_level_3 = np.count_nonzero(np.where(np.abs(deviation_in_sigmas) < 3., True, False)) / total * 100  # 99.7%
    sigma_level_greater_3 = np.count_nonzero(np.where(np.abs(deviation_in_sigmas) > 3., True, False)) / total * 100  # 0.3%
    sigma_level_greater_5 = np.count_nonzero(np.where(np.abs(deviation_in_sigmas) > 5., True, False)) / total * 100  # 0.0%

    ax.text(0.02, 0.98, r'$z_{phot}$' + ' outlier levels with\n expected values in brackets:'
            + '\n' + r'$<1 \sigma =$' + '{:.1f}% (68.3%)'.format(sigma_level_1)
            + '\n' + r'$<2 \sigma =$' + '{:.1f}% (95.5%)'.format(sigma_level_2)
            + '\n' + r'$<3 \sigma =$' + '{:.1f}% (99.7%)'.format(sigma_level_3)
            + '\n' + r'$>3 \sigma =$' + '{:.1f}% (0.3%)'.format(sigma_level_greater_3)
            + '\n' + r'$>5 \sigma =$' + '{:.1f}% (0.0%)'.format(sigma_level_greater_5)
            + '\n' + 'N = {}'.format(total),
            ha='left', va='top', transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),))

    # Limits and labels
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
    ax.set_xlabel(r'$(z_{phot} - z_{spec}) / \sigma_{z_{phot}}$')
    ax.set_ylabel('Probability')
    ax.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)

    # Output time
    if plt_title is not None:
        ax.set_title(plt_title)

    if save_name is not None:
        fig.savefig(save_name)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def error_evaluator_wittman(spectroscopic_z, validation_mixtures, validation_results,
                            plt_title: Optional[str]=None, save_name: Optional[str]=None, show_fig: bool=False,
                            validity_condition: str='greater_than_zero'):
    """Makes a histogram evaluating how good the given redshift error estimates are, using the Wittman+2014 method of
    sorting and summing cdfs instead of using sigma outliers and frequentism.

    CURRENTLY ONLY WORKS WITH NORMAL DISTRIBUTIONS.

    Args:
        Function-specific:
            spectroscopic_z (array-like): true redshifts to test against.
            photometric_z (array-like): inferred redshifts to test.
            sigma_1_lower (array-like): lower bounds on redshifts. Should be a positive
            sigma_1_upper (array-like): upper bounds on redshifts. Set to None, this will be a copy of sigma_1_low (useful
                for if error estimates are symmetrical.)
            limits (array-like): x limits for the plot.
            # todo rewrite this shite

        Module-common:
            plt_title (None or bool): title for the plot.
            save_name (None or str): name of plot to save.
            show_fig (bool): whether or not to show the plot.
            validity_condition (str): validity condition to pass to util.where_valid_redshifts().

    Returns:
        A gorgeous plot
    """
    # Just in case you passed a data frame or some bullshit like that
    spectroscopic_z = np.asarray(spectroscopic_z).flatten()

    # Calculate cdfs of all mixtures at all ma points!
    cdfs = scipy_normal.cdf(np.tile(spectroscopic_z, (validation_mixtures['means'].shape[1], 1)).T,
                            loc=validation_mixtures['means'],
                            scale=validation_mixtures['std_deviations'])

    # Multiply by the weights and sum
    cdfs = np.sum(validation_mixtures['weights'] * cdfs, axis=1)

    # Normalise the cdfs with the initial and end values
    cdfs = cdfs * validation_results['cdf_multiplier'] + validation_results['cdf_constant']

    cdfs_sorted = np.sort(cdfs)

    # Cumulative sum and normalise so last element is 1.0
    # cdfs_summed = np.cumsum(cdfs)
    # cdfs_summed /= 0.5 * cdfs_summed.size

    cdfs_summed = np.linspace(0., 1., num=cdfs_sorted.size)

    residuals = (cdfs_summed - cdfs_sorted)**2
    mean_residual = np.mean(residuals)
    max_residual = np.max(residuals)

    # Make a Wittman+2014-style plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(cdfs_sorted, cdfs_summed, 'r-')

    ax.text(0.02, 0.98, 'Mean residual = {:.5f}'.format(mean_residual)
            + '\nMax residual = {:.5f}'.format(max_residual),
            ha='left', va='top', transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0), ))

    # Labels
    ax.set_xlabel(r'$c_i$')
    ax.set_ylabel(r'$F(c_i)$')

    # Output time
    if plt_title is not None:
        ax.set_title(plt_title)

    if save_name is not None:
        fig.savefig(save_name)

    if show_fig:
        fig.show()
    else:
        plt.close(fig)

    return [mean_residual, max_residual]


"""A set of unit tests and stuff to let us run this script independently for plots if desired."""
if __name__ == '__main__':
    print('Script is being ran independently! Beginning unit tests:')

    # Variables for unit test
    data_dir = './data/KMOS415_output/'  # Reminder: put a slash on the end
    files_to_read = ['GS415.3dhst.redshift.save', 'GS415.coo.save']

    # Read in the .save files
    print('Reading in default save files')
    redshifts = scripts.file_handling.read_save(data_dir + files_to_read[0])
    coords = scripts.file_handling.read_save(data_dir + files_to_read[1])

    # Join the co-ordinates to the redshifts DataFrame
    redshifts['gs4_ra'] = coords['gs4_ra']
    redshifts['gs4_dec'] = coords['gs4_dec']

    # Calculate the NMAD
    my_nmad = util.calculate_nmad(redshifts['gs4_zspec'], redshifts['gs4_zphot'])

    # Make a plot of the photometric redshifts against spectroscopic
    phot_vs_spec(redshifts['gs4_zspec'], redshifts['gs4_zphot'], show_nmad=True, show_fig=True, limits=[0, 7])

    # Remove everything within specific sky co-ordinates
    #redshifts = redshifts.iloc[redshifts.gs4_ra.values < 53.1]

    # Make a plot of locations on the sky
    #sky_locations(redshifts['gs4_ra'], redshifts['gs4_dec'])

    # Find all galaxy pairs
    random_catalogue_repeats = 1
    #all_galaxy_pairs, random_galaxy_pairs = scripts.galaxy_pairs.store_pairs_on_sky(redshifts['gs4_ra'][:],
    #                                                                  redshifts['gs4_dec'][:],
    #                                                                                max_separation=15., min_separation=1.5,
    #                                                                                max_move=26, min_move=25,
    #                                                                                size_of_random_catalogue=random_catalogue_repeats)
    # Try reading in the pairs again to check the storing worked
    max_z = 100.0
    min_z = 0.0
    all_galaxy_pairs_read_in = scripts.galaxy_pairs.read_pairs('./data/all_pairs.csv', redshifts['gs4_zphot'],
                                                               min_redshift=min_z, max_redshift=max_z)

    random_galaxy_pairs_read_in = scripts.galaxy_pairs.read_pairs('./data/random_pairs.csv', redshifts['gs4_zphot'],
                                                                  min_redshift=min_z, max_redshift=max_z,
                                                                  size_of_random_catalogue=random_catalogue_repeats)

    # Make a plot of Npairs against deltaZ
    #pair_redshift_deviation(redshifts['gs4_zphot'], all_galaxy_pairs_read_in, random_galaxy_pairs_read_in,
    #                        size_of_random_catalogue=random_catalogue_repeats, show_fig=True)

    # Test how good or bad the error estimates are
    error_evaluator(redshifts['gs4_zspec'], redshifts['gs4_zphot'],
                    redshifts['gs4_zphot_l68'], redshifts['gs4_zphot_u68'], show_fig=True, limits=[-5, 5])
