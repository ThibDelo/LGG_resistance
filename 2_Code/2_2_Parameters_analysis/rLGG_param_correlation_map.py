
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Correlation plot

# Author: Thibault Delobel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as pl
from matplotlib.colors import Normalize


p = plt.rcParams
p['font.size'] = 15
p.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
p.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})


df = pd.read_excel('1_Data/param_model_v5.xlsx')
df.index = df['Patient']
df = df.drop('Patient', axis=1)
df = df.drop('lambda1', axis=1)
df = df.drop('beta', axis=1)

# virtual dataset
df2 = pd.read_csv('1_Data/param_clinical_virutal_study.csv', sep=',')
df2 = df2.drop('l1', axis=1)
df2 = df2.drop('b', axis=1)
df2 = df2.drop('lamb', axis=1)

param_list = [r'$\rho_1$', r'$\rho_2$', r'$\tau$',  r'$\psi$', r'$\alpha_1$', r'$\alpha_2$']

norm = Normalize(vmin=0, vmax=1)

fig1, ax = plt.subplots(figsize=(8,7))
p["ytick.minor.visible"] = True
sns.heatmap(df2.corr(method='pearson'), vmin=0, vmax=1, annot=True, cmap='YlGnBu', cbar=False)
cbar = fig1.colorbar(pl.cm.ScalarMappable(norm=norm, cmap='YlGnBu'), ax=ax)
p["ytick.minor.visible"] = False
cbar.set_label('Pearson~correlation~coefficient')
plt.xticks(np.arange(0.5, 6.5, 1), param_list)
plt.yticks(np.arange(0.5, 6.5, 1), param_list, rotation=0)
plt.title('Virtual parameters correlation')
plt.tight_layout()
fig1.savefig('3_Plots/3_2_Param_correlation/corr_pearson_matrix_virtual_study.png', dpi=300)
plt.show()


fig2, ax = plt.subplots(figsize=(8,7))
sns.heatmap(df.corr(method='spearman'), vmin=0, vmax=1, annot=True, cmap='YlGnBu', cbar=False)
p["ytick.minor.visible"] = True
cbar = fig2.colorbar(pl.cm.ScalarMappable(norm=norm, cmap='YlGnBu'), ax=ax)
cbar.set_label('Spearman~correlation~coefficient')
p["ytick.minor.visible"] = False
plt.title('Real parameters correlation')
plt.xticks(np.arange(0.5, 6.5, 1), param_list)
plt.yticks(np.arange(0.5, 6.5, 1), param_list, rotation=0)
plt.tight_layout()
fig2.savefig('3_Plots/3_2_Param_correlation/corr_spearman_matrix_real_patient.png', dpi=300)
plt.show()
