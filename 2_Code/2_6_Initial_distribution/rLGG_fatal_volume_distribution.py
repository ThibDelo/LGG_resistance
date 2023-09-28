
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Distribution of fatal volume

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 23/01/2023


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_plots import custom_plt_param

# change defaut matplolib param
plt.rcParams = custom_plt_param

# generate the distribution
x = np.linspace(200, 360, 10000)
dist = scipy.stats.norm.pdf(x, 280, 20)

# plot
plt.figure(figsize=(6, 6))
plt.plot(x, dist, color='k', label='Estimated gaussian distribution', lw=2)
plt.ylim([0, 0.025])
plt.xlim([200, 360])
plt.legend()
plt.xlabel('Fatal~tumor~volume~($cm^3$)')
plt.ylabel('Density')
plt.title('Fatal volume distribution')
plt.tight_layout()
plt.savefig('3_Plots/3_5_Initial_distribution/fatal_volume_plot.png', dpi=300)
plt.show()
