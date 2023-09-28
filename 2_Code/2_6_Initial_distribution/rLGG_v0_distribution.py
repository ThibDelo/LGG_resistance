
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Distribution of initial tumor volume

# Author: Thibault Delobel
# creation: 08/08/2022
# last edit: 23/01/2023


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import sys

sys.path.append('2_Code/')

from rLGG_package.rLGG_plots import custom_plt_param

# change defaut matplolib param
plt.rcParams = custom_plt_param

# data
df = pd.read_csv('1_Data/initial_size_chemo.csv')
X = np.array(df['Volume']).reshape(-1, 1)

# fit data distribution with a gaussian kernel density estimation
X_plot = np.linspace(min(X), max(X), 1000)
kde = KernelDensity(kernel="gaussian", bandwidth=10).fit(X)
log_dens = kde.score_samples(X_plot)

# plot
plt.figure(figsize=(6, 6))
plt.plot(X_plot[:, 0], np.exp(log_dens), lw=2, linestyle="-", label='Empirical distribution', c='k')
plt.hist(X, density=1, label='Real distribution (n=37)', color='#d1edb3', bins=9)
plt.xlim([min(X), max(X)])
plt.ylim([0, 0.015])
plt.xlabel('Initial~tumor~volume~($cm^3$)')
plt.ylabel('Density')
plt.title('Initial volume distribution')
plt.legend()
plt.tight_layout()
plt.savefig('3_Plots/3_5_Initial_distribution/initial_size_density_plot.png', dpi=300)
plt.show()
