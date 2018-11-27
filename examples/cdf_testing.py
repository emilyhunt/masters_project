import matplotlib.pyplot as plt
import numpy as np
from scripts import loss_funcs
from scipy.stats import norm as scipy_normal

# Grab some normal points
distribution = loss_funcs.NormalPDFLoss()

hello = {}
to_try = np.asarray([50, 100, 500])
colours = ['r', 'g', 'b', 'p']

for i, c in zip(to_try, colours):
    random_variables = scipy_normal.rvs(size=i, loc=0., scale=1.)

    # Evaluate the cdf at each random deviate and sort the array
    cdfs = scipy_normal.cdf(random_variables, loc=1.0, scale=0.5)

    cdfs_sorted = np.sort(cdfs)

    # Cumulative sum and normalise so last element is 1.0
    cdfs_summed = np.cumsum(cdfs)
    cdfs_summed /= cdfs_summed[-1]

    # Get expected points
    cdfs_expected = np.linspace(0., 1., num=i)

    cdfs_summed = (cdfs_summed - cdfs_sorted)

    # Plot shiz
    plt.plot(cdfs_sorted, cdfs_summed, '-', lw=1, markerfacecolor='c', ms=2, label=i)
    plt.xlabel('ci')
    plt.ylabel('F(ci) - ci')

boop = np.mean(cdfs_summed**2)
plt.plot([0, 1], [boop, boop], 'm--')
plt.plot([0, 1], [0, 0], 'k--')
plt.legend()
plt.show()

