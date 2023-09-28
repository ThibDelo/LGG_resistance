
# -----------------------------------------------
# Overcoming chemotherapy resistance in low-grade
# gliomas: A computational approach
# -----------------------------------------------

# Linear regression plots

# Author: Thibault Delobel

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


p = plt.rcParams
p['font.size'] = 14

p.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
p.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})

p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.linestyle"] = (0, (15, 10))
p["grid.color"] = "#B6BBBF"
p["grid.linewidth"] = 0.25


def reg_line(a, b, min_x, max_x):
    x = np.linspace(min_x, max_x, 1000)
    y = a*x + b
    plt.plot(x, y, c='k', lw=2, ls='--')


# Data importaion -----------------------------------------

# real patients dataset
df = pd.read_excel('1_Data/param_model_v5.xlsx')
df.index = df['Patient']
df = df.drop('Patient', axis=1)
df = df.drop('lambda1', axis=1)
df = df.drop('beta', axis=1)

# virtual patients dataset
df2 = pd.read_csv('1_Data/param_clinical_virutal_study.csv', sep=',')
df2 = df2.drop('l1', axis=1)
df2 = df2.drop('b', axis=1)
df2 = df2.drop('lamb', axis=1)


# stats --------------------------------------------------

slope, intercept, r_value, p_value, std_err = stats.linregress(df2['a3'], df2['a4'])

# plot ----------------------------------------------------

# virtual patient fig
plt.figure(figsize=(14, 5))
plt.suptitle('Linear regression model of correlated parameters of virtual patients')

plt.subplot(131)
reg_line(0.629, 0.0036, min(df2['rho3']), max(df2['rho3']))
plt.scatter(df2['rho3'], df2['rho2'], c='#D1EDB3')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho_2$')

plt.subplot(132)
reg_line(613, 0.1, min(df2['rho1']), max(df2['rho1']))
plt.scatter(df2['rho1'], df2['a2'], c='#D1EDB3')
plt.xlabel(r'$\rho_1$')
plt.ylabel(r'$\psi$')

plt.subplot(133)
reg_line(0.161, 0.027, min(df2['a3']), max(df2['a3']))
plt.scatter(df2['a3'], df2['a4'], c='#D1EDB3')
plt.xlabel(r'$\alpha_1$')
plt.ylabel(r'$\alpha_2$')

plt.tight_layout()
plt.savefig('3_Plots/3_2_Param_correlation/reg_virtual_patients.png', dpi=600)
plt.show()


# real patient fig
plt.figure(figsize=(14, 5))
plt.suptitle('Linear regression model of correlated parameters of real patients')

plt.subplot(131)
reg_line(0.749, 0.0008, min(df2['rho3']), max(df2['rho3']))
plt.scatter(df['rho3'], df['rho2'], c='#D1EDB3')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho_2$')

plt.subplot(132)
reg_line(722, -0.1834, min(df2['rho1']), max(df2['rho1']))
plt.scatter(df['rho1'], df['alpha2'], c='#D1EDB3')
plt.xlabel(r'$\rho_1$')
plt.ylabel(r'$\psi$')

plt.subplot(133)
reg_line(0.168, 0.0074, min(df2['a3']), max(df2['a3']))
plt.scatter(df['alpha3'], df['alpha4'], c='#D1EDB3')
plt.xlabel(r'$\alpha_1$')
plt.ylabel(r'$\alpha_2$')

plt.tight_layout()
plt.savefig('3_Plots/3_2_Param_correlation/reg_real_patients.png', dpi=600)
plt.show()

