"""Makes an SFR vs MStar density plot for the final report."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Make TeX labels work on plots
#plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})
#plt.rc('text', usetex=True)

# Constants and shit
grid_resolution = 20
n_levels = 50
n_shape = 60
x_lim = [-10, 20]
y_lim = [-10, 20]
z_lim = [-2000, 50000]


# DATA READING
validation_no_spec_z = pd.read_csv('./final_run_data/validation_results_no_spec_z.csv')
data_no_spec_z = pd.read_csv('./final_run_data/data_no_spec_z.csv')
data_train = pd.read_csv('./final_run_data/data_validation.csv')

x = data_no_spec_z['sed_log_mass'].values
y = data_no_spec_z['sed_log_sfr'].values
z = np.asarray(np.abs(validation_no_spec_z['map'] - data_no_spec_z['z_phot_lit'])
        / (1 + 0.5*(validation_no_spec_z['map'] + data_no_spec_z['z_phot_lit'])))

# SET LIMITS OF TEST DATA
# Set resolution
x_resolution = y_resolution = grid_resolution

# Setup the limits, ignoring anything not within said limits
allowed_x = np.where(np.logical_and(x_lim[0] < x, x < x_lim[1]), True, False)
allowed_y = np.where(np.logical_and(y_lim[0] < y, y < y_lim[1]), True, False)
allowed_z = np.where(np.logical_and(z_lim[0] < z, z < z_lim[1]), True, False)

# Now, only keep values that satisfy all conditions
allowed_values = np.logical_and(allowed_x, np.logical_and(allowed_y, allowed_z))
x = x[allowed_values]
y = y[allowed_values]
z = z[allowed_values]


# TRAINING DATA LIMIT FINDING
# Find the limits of the data
x_train = data_train['sed_log_mass'].values
y_train = data_train['sed_log_sfr'].values

allowed_x = np.where(np.logical_and(x_lim[0] < x_train, x_train < x_lim[1]), True, False)
allowed_y = np.where(np.logical_and(y_lim[0] < y_train, y_train < y_lim[1]), True, False)
allowed_values = np.logical_and(allowed_x, allowed_y)

x_train = x_train[allowed_values]
y_train = y_train[allowed_values]

# Make an arbitrary shape of radius 1, then convert from polar coords back to cartesian
# interior_angle = (n_shape - 2) * np.pi / n_shape
angles = np.linspace(0, 2 * np.pi - (2 * np.pi / n_shape), num=n_shape)
shape_x = np.cos(angles)
shape_y = np.sin(angles)

# Normalise the shape coords to be in the range [0, 1]
shape_x = shape_x / (shape_x.max() - shape_x.min())
shape_x -= shape_x.min()
shape_y = shape_y / (shape_y.max() - shape_y.min())
shape_y -= shape_y.min()

# Stretch the shape to fit across the data
shape_x = (x_train.max() - x_train.min()) * shape_x + x_train.min()
shape_y = (y_train.max() - y_train.min()) * shape_y + y_train.min()

# Make a big array of distances from each point of each point, then find the nearest one each time
distance_from_points = np.sqrt((shape_x.reshape(-1, 1) - x_train)**2 + (shape_y.reshape(-1, 1) - y_train)**2)
point_ids = np.argmin(distance_from_points, axis=1)
#point_ids = np.unique(point_ids)

# Grab the corresponding points to this from x and y train, and add the end on too so we plot a complete shape
x_train_limits = x_train[point_ids]
y_train_limits = y_train[point_ids]
x_train_limits = np.append(x_train_limits, x_train_limits[0])
y_train_limits = np.append(y_train_limits, y_train_limits[0])


# PLOTTING SHIT - TEST DATA
# Calculate some ranges to grid over
x_range = np.linspace(x.min(), x.max(), x_resolution)
x_spacing = np.abs(x_range[1] - x_range[0])
y_range = np.linspace(y.min(), y.max(), y_resolution)
y_spacing = np.abs(y_range[1] - y_range[0])

# Make grid points for later
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Cycle over the different grid points calculating the mean redshift in each place
mean_z = np.zeros(x_grid.shape)
training_data_density = np.zeros(x_grid.shape)
i = 0
for a_x in x_range:
    j = 0
    for a_y in y_range:
        good_x = np.logical_and(a_x - 0.5*x_spacing < x, x < a_x + 0.5*x_spacing)
        good_y = np.logical_and(a_y - 0.5*y_spacing < y, y < a_y + 0.5*y_spacing)
        good_z = z[np.logical_and(good_x, good_y)]

        good_x_train = np.logical_and(a_x - 0.5 * x_spacing < x_train, x_train < a_x + 0.5 * x_spacing)
        good_y_train = np.logical_and(a_y - 0.5 * y_spacing < y_train, y_train < a_y + 0.5 * y_spacing)
        training_data_density[i, j] = np.count_nonzero(np.logical_and(good_x_train, good_y_train))

        # Only calculate the mean if there's more than one in this bin!
        if good_z.size > 0:
            mean_z[i, j] = np.mean(good_z)
        else:
            mean_z[i, j] = np.nan
        j += 1
    i += 1


# PLOTTING TIME
# The test data
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(1, 1, 1)
#density_plot_object = ax.contourf(x_grid, y_grid, mean_z.T, n_levels, cmap=plt.get_cmap('plasma'), corner_mask=False)
density_plot_object = ax.imshow(np.flip(mean_z.T, axis=0),
                                extent=[x.min() - 0.5*x_spacing, x.max() + 0.5*x_spacing,
                                        y.min() - 0.5*y_spacing, y.max() + 0.5*y_spacing],
                                cmap=plt.get_cmap('coolwarm'), alpha=1)

# Now, plot the training data's limits
ax.contour(x_grid, y_grid, np.log(training_data_density.T + 1), 5, colors='k', linewidths=[2, 1, 1, 1, 1])
#ax.plot(x_train_limits, y_train_limits, 'k-', label='Training data extent')
#ax.plot(shape_x, shape_y, 'b-', label='Catching shape')
#ax.plot(x_train, y_train, 'k.', alpha=0.1, ms=2)

# And let's label what the contours are
#ax.text(1, 3, 'Training data\nnumber density', fontsize=8,
#        bbox=dict(boxstyle='round', ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),))
#ax.arrow(6, 1.5, 1, -0.5, head_width=0.3, head_length=0.3)


# Setup of the colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
colorbar = fig.colorbar(density_plot_object, cax=cax, orientation='vertical')
colorbar.set_label(r'$(z_{EAZY} - z_{ML}) \: / \: (1 + z_{EAZY})$')

# Labels and legend
ax.set_xlabel(r'$\log (M_{stars})$' + ' (stellar mass)')
ax.set_ylabel(r'$\log (SFR)$' + ' (star formation rate)')
#ax.legend(edgecolor='k', facecolor='w', fancybox=True, fontsize=8)

#fig.tight_layout()
fig.savefig('./final_plots/pres_2_sfr_vs_mass.png', dpi=600, bbox_inches='tight')
fig.show()
