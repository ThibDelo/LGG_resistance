
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Distribution of TMZ cycle number

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 23/01/2023

import sys
import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import cycler
import matplotlib.pyplot as plt

sys.path.append('2_Code/')

from rLGG_package.rLGG_plots import custom_plt_param

# change defaut matplolib param
plt.rcParams = custom_plt_param

# create distribution
rdist = [124, 80, 74, 135, 40, 130]
x = np.linspace(5, 33, 1000)
dist = scipy.stats.norm.pdf(x, 19, 7)
x = np.digitize(x, list(range(5, 34, 1)))
x = x + 4

# plot
plt.figure(figsize=(6, 6))
plt.plot(x, dist, lw=2, c='k', label='Estimated gaussian distribution')
plt.xlim([5, 33])
plt.ylim([0, 0.07])
plt.legend()
plt.xlabel('Number~of~TMZ~cycle')
plt.ylabel('Density')
plt.title('TMZ cycles distribution')
plt.tight_layout()
plt.savefig('3_Plots/3_5_Initial_distribution/number_cycle_density_plot.png', dpi=300)
plt.show()
