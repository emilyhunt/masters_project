import matplotlib.pyplot as plt
import numpy as np
from scripts import loss_funcs
from scipy.stats import norm as scipy_normal

# Grab some normal points
distribution = loss_funcs.NormalPDFLoss()

hello = {}
to_try = np.asarray([1500] * 1)
#colours = ['r', 'g', 'b', 'm', 'c', 'y']

mean = [0.0, 0.1, 0.2, 0.3, 0.4]
mean = [0.0, 0.0, 0.0, 0.0, 0.0]
std_d = [1.0, 1.1, 1.2, 1.3, 1.4]
colors = ['r', 'g', 'c', 'b', 'm']

means = np.zeros((len(mean), to_try.size))

for stage_i, (a_mean, a_std_d, a_color) in enumerate(zip(mean, std_d, colors)):
    for try_i, i in enumerate(to_try):
        random_variables = scipy_normal.rvs(size=i, loc=0., scale=1.)

        # Evaluate the cdf at each random deviate and sort the array
        cdfs = scipy_normal.cdf(random_variables, loc=0.0, scale=a_std_d)

        cdfs = cdfs[np.where(np.logical_and(cdfs > a_mean, cdfs < 1 - a_mean))[0]]

        # Make it median-invariant
        #cdfs = np.abs(cdfs - 0.5)
        cdfs_sorted = np.sort(cdfs)

        # Extend the cdfs and take means
        #cdfs[:] = 0.5
        #np.random.shuffle(cdfs)
        #cdfs = np.mean(cdfs.reshape(random_variables.shape[0], -1), axis=1)

        # Cumulative sum and normalise so last element is 1.0
        cdfs_summed = np.cumsum(cdfs)
        cdfs_summed /= 0.5 * i

        # Get expected points
        cdfs_expected = np.linspace(0., 1., num=cdfs_sorted.size)
        cdfs_summed = cdfs_expected

        cdfs_summed = np.log(np.cosh(cdfs_summed - cdfs_sorted))

        # Plot shiz
        plt.plot(cdfs_sorted, cdfs_summed, '-', lw=1, ms=2, color=a_color, alpha=0.1)
        boop = np.max(cdfs_summed)
        #plt.plot([0, 1], [boop, boop], '--', color=c, label='mean^2 of {}'.format(i))

        means[stage_i, try_i] = boop

    boop = np.mean(means[stage_i, :])
    plt.plot([0, 1], [boop, boop], '--', color=a_color, label='mean^2 of {}, {}'.format(a_mean, a_std_d))


#plt.plot([0, 1], [0, 0], 'k--')
plt.legend()
plt.xlabel('ci')
plt.ylabel('(F(ci) - ci)^2')
#plt.title('CDF stat for mean {}, std_d {} of model'.format(mean, std_d))
plt.show()

print("Std deviation in means: {}".format(np.std(means, axis=1)))

