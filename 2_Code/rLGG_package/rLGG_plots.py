
import matplotlib.pyplot as plt
from matplotlib import cycler
import numpy as np


# change default matplotlib style
custom_plt_param = plt.rcParams
custom_plt_param["ytick.minor.visible"] = True
custom_plt_param["xtick.minor.visible"] = True
custom_plt_param["axes.grid"] = True
custom_plt_param["grid.linestyle"] = (0, (15, 10))
custom_plt_param["grid.color"] = "#B6BBBF"
custom_plt_param["grid.linewidth"] = 0.25
custom_plt_param['font.size'] = 14
custom_plt_param["axes.prop_cycle"] = cycler(color=['#1f7bb6', '#7cc499', '#F0C929', 
                                     '#F48B29', '#AC0D0D']) 
custom_plt_param.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
custom_plt_param.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})


def plot_fit(t, Y, data_time, data_volume, treatment, patient_id, 
             ylim=None, dpi=300, format_img='png', show_img=True, 
             save_img=True, save_path='auto', errorbar=True):

    volume = Y[:, 0] + Y[:, 1] + Y[:, 2] + Y[:, 3] + Y[:, 4]
    time_critical_volume = t[np.where(volume > 280)][0]/30
    
    plt.figure(figsize=(10, 6))
    #-- plot tumor volume from simulaton 
    plt.plot(t/30, volume, c='k', label='model', lw=1.5, zorder=5)
    #-- plot real patient data
    if errorbar:
        plt.errorbar(data_time/30, data_volume, yerr=data_volume*0.2, fmt="o", 
                     ecolor='k', c='w', label='data', markeredgecolor='k',
                     markeredgewidth=2, zorder=10)
    else:
        pass
    #-- plot each cell population volume from simulation
    plt.plot(t/30, Y[:, 0], label='Vs', lw=1.5)
    plt.plot(t/30, Y[:, 1], label='Vd', lw=1.5)
    plt.plot(t/30, Y[:, 2], label='VPi', lw=1.5)
    plt.plot(t/30, Y[:, 3], label='VP', lw=1.5)
    plt.plot(t/30, Y[:, 4], label='Vr', lw=1.5)
    #-- plot treatments
    for key in treatment.keys():
        # chemotherapy
        if key[0] == 'c':
            plt.axvspan(treatment[key][0]/30, treatment[key][1]/30, 
                        color='#eff9b6', alpha=0.5, label='chemotherapy') 
        # radiotherapy
        if key[0] == 'r':
            plt.axvspan(treatment[key][0]/30, treatment[key][1]/30, 
                        color='#f5c9c9', alpha=1, label='radiotherapy')   
    #-- decoration
    plt.xlabel('Time~(months)')
    plt.ylabel('Volume~($cm^3$)')
    if ylim != None:
        plt.ylim([0, ylim])
    else: 
        pass
    plt.xlim([0, time_critical_volume])
    plt.title(f'Patient {patient_id}')
    plt.tight_layout()
    #-- show and save image
    if save_img == True and save_path == 'auto':
        plt.savefig(f'3_Plots/3_1_Fits/fit_p{patient_id}.{format_img}', 
                    dpi=dpi)
    elif save_img == True and save_path != 'auto':
        plt.savefig(save_path, dpi=dpi) 
    else:
        pass
    if show_img == True:
        plt.show()
    else:
        plt.close()
