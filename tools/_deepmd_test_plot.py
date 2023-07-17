#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Very fancy DeepMD-based semi-automatic highly-customizable iterative training procedure. #
#   Copyright 2022-2023 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

import numpy as np
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import max_error as MAXE

deepmd_iterative_apath = Path("_DEEPMD_ITERATIVE_APATH_")
sys.path.insert(0, str(deepmd_iterative_apath/"tools"))
import common_functions as cf
import _plot_functions as pf
training_iterative_apath = Path("..").resolve()

### Read what is needed (json files)
control_apath = training_iterative_apath/"control"
config_json = cf.json_read((control_apath/"config.json"),True,True)
current_iteration_zfill = Path().resolve().parts[-1].split('-')[0]
current_iteration = int(current_iteration_zfill)
test_json = cf.json_read((control_apath/("test_"+current_iteration_zfill+".json")),True,True)
current_apath = Path(".").resolve()

energy_f = np.load(str(current_apath/"energy_sys.npz"),allow_pickle=True)
force_f = np.load(str(current_apath/"force_sys.npz"),allow_pickle=True)

energy,force={},{}
for f in energy_f.files:
    energy[f] = energy_f[f].tolist()
    force[f] = force_f[f].tolist()
del energy_f, force_f
gc.collect()

fig_apath = training_iterative_apath/"figures"/"test"/(current_iteration_zfill+"-details")

Path("../figures/test/"+current_iteration_zfill+"-details")
fig_apath.mkdir(exist_ok=True, parents=True)

dpi = 300.0
sizemult = 1.0
size = 16.0
ratio = 8.0/3.0
figsize_ratio = np.array((size,size/ratio))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
linestyles = ["-","--",":"]

props = dict(boxstyle="round", facecolor="grey", alpha=0.2)

test_json["results"]={}
for f in energy.keys():
    test_json["results"][f]={}
    for g in energy[f].keys():
        test_json["results"][f][g]={}
        fig,ax=plt.subplots(nrows=1,ncols=2,dpi=dpi/2,figsize=figsize_ratio*sizemult)
        print(f,g,flush=True)
        gc.collect()
        title="NNP: "+current_iteration_zfill+"-"+f+" System: "+g+" "+str(energy[f][g].shape[0])
        nb_atom=int(force[f][g].shape[0]/energy[f][g].shape[0]/3)

        ax=plt.subplot(1,2,1)
        pf.axes(ax,sizemult)
        if energy[f][g].ndim > 1:
            min_ref=np.min(energy[f][g][:,0])
            min_all=np.min((np.min(energy[f][g][:,0]),np.min(energy[f][g][:,1])))
            max_all=np.max((np.max(energy[f][g][:,0]),np.max(energy[f][g][:,1])))
            MAE_val=np.round(MAE(energy[f][g][:,0]/nb_atom,energy[f][g][:,1]/nb_atom),10)
            MSE_val=np.round(MSE(energy[f][g][:,0]/nb_atom,energy[f][g][:,1]/nb_atom),10)
            RMSE_val=np.round(MSE(energy[f][g][:,0]/nb_atom,energy[f][g][:,1]/nb_atom,squared=False),10)
            MAXE_val=np.round(MAXE(energy[f][g][:,0]/nb_atom,energy[f][g][:,1]/nb_atom),10)
            RMSErel_val = np.round( ( RMSE_val / np.std(energy[f][g][0]) ), 10 )
        else:
            min_ref=np.min(energy[f][g][0])
            min_all=np.min((np.min(energy[f][g][0]),np.min(energy[f][g][1])))
            max_all=np.max((np.max(energy[f][g][0]),np.max(energy[f][g][1])))
            MAE_val=np.round(MAE(energy[f][g][0]/nb_atom,energy[f][g][1]/nb_atom),10)
            MSE_val=np.round(MSE(energy[f][g][0]/nb_atom,energy[f][g][1]/nb_atom),10)
            RMSE_val=np.round(MSE(energy[f][g][0]/nb_atom,energy[f][g][1]/nb_atom,squared=False),10)
            MAXE_val=np.round(MAXE(energy[f][g][0]/nb_atom,energy[f][g][1]/nb_atom),10)
            RMSErel_val = np.round( ( RMSE_val / np.std(energy[f][g][0]) ), 10)

        textstr = "\n".join((
        r"MAE = %.2e" % (MAE_val, )+ r" eV/atom",
        r"MSE = %.2e" % (MSE_val, )+ r" eV/atom",
        r"RMSE = %.2e" % (RMSE_val, )+ r" eV/atom",
        r"RMSE / σ = %.2e" % (RMSErel_val, ),
        r"MAXE = %.2e" % (MAXE_val, )+ r" eV/atom",))
        test_json["results"][f][g]["MAE_e_atm"] = MAE_val
        test_json["results"][f][g]["MSE_e_atm"] = MSE_val
        test_json["results"][f][g]["RMSE_e_atm"] = RMSE_val
        test_json["results"][f][g]["MAXE_e_atm"] = MAXE_val
        test_json["results"][f][g]["RMSE_rel_e_atm"] = RMSErel_val
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=sizemult*14,
            verticalalignment="top",bbox=props)
        if energy[f][g].ndim > 1:
            ax.plot(energy[f][g][:,0]-min_ref,energy[f][g][:,1]-min_ref,linestyle="None",marker="x",color=colors[0])
        else:
            ax.plot(energy[f][g][0]-min_ref,energy[f][g][1]-min_ref,linestyle="None",marker="x",color=colors[0])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,color=colors[1])
        inter=int(pf.anint(((max_all-min_ref)-(min_all-min_ref))/5))
        pf.xaxe(ax,int(pf.anint((min_all-min_ref))),int(pf.anint((max_all-min_ref))),inter,0,"{0:.0f}",sizemult)
        ax.axes.set_xlabel(r"REF Energies [eV]",{"fontsize":sizemult*25,"fontweight":"normal"})
        pf.yaxe(ax,int(pf.anint((min_all-min_ref))),int(pf.anint((max_all-min_ref))),inter,0,"{0:.0f}",sizemult)
        ax.axes.set_ylabel(r"NNP Energies [eV]",{"fontsize":sizemult*25,"fontweight":"normal"})

        ax=plt.subplot(1,2,2)
        pf.axes(ax,sizemult)
        min_all=np.min((np.min(force[f][g][:,0]),np.min(force[f][g][:,1])))
        max_all=np.max((np.max(force[f][g][:,0]),np.max(force[f][g][:,1])))
        max_uall=np.ceil(np.max((np.abs(min_all),np.abs(max_all))))
        MAE_val=np.round(MAE(force[f][g][:,0],force[f][g][:,1]),3)
        MSE_val=np.round(MSE(force[f][g][:,0],force[f][g][:,1]),3)
        RMSE_val=np.round(MSE(force[f][g][:,0],force[f][g][:,1],squared=False),3)
        MAXE_val=np.round(MAXE(force[f][g][:,0],force[f][g][:,1]),3)
        RMSErel_val = np.round( ( RMSE_val / np.std(force[f][g][:,0]) ),3)

        textstr = "\n".join((
        r"MAE = %.2e" % (MAE_val, )+ r" eV/Å",
        r"MSE = %.2e" % (MSE_val, )+ r" eV/Å",
        r"RMSE = %.2e" % (RMSE_val, )+ r" eV/Å",
        r"RMSE / σ = %.2e" % (RMSErel_val, ),
        r"MAXE = %.2e" % (MAXE_val, )+ r" eV/atom",))
        test_json["results"][f][g]["MAE_f"] = MAE_val
        test_json["results"][f][g]["MSE_f"] = MSE_val
        test_json["results"][f][g]["RMSE_f"] = RMSE_val
        test_json["results"][f][g]["MAXE_f"] = MAXE_val
        test_json["results"][f][g]["RMSE_rel_f"] = RMSErel_val
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=sizemult*14,
            verticalalignment="top",bbox=props)
        ax.plot(force[f][g][:,0],force[f][g][:,1],linestyle="None",marker="x",color=colors[0])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,color=colors[1])
        ax.plot([0.5, 0.5], [1, 0], transform=ax.transAxes,color="black",linestyle="--",label="")
        ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes,color="black",linestyle="--",label="")
        range_val=pf.max_range(max_uall)
        pf.xaxe(ax,-range_val,range_val,range_val/5,0,"{: 5.0f}",sizemult)
        ax.axes.set_xlabel(r"REF Forces [eV/Å]",{"fontsize":sizemult*25,"fontweight":"normal"})
        pf.yaxe(ax,-range_val,range_val,range_val/5,0,"{: 5.0f}",sizemult)
        ax.axes.set_ylabel(r"NNP Forces [eV/Å]",{"fontsize":sizemult*25,"fontweight":"normal"})
        plt.tight_layout(pad=3.0)
        fig.suptitle(title,fontsize=sizemult*25,fontweight="normal",y=1.025)
        name=current_iteration_zfill+"_"+g+"_"+f
        fig.savefig(str(fig_apath/(name+".png")),dpi=300,bbox_inches="tight",transparent=True)
        del ax, fig, name
        del min_all, max_all, max_uall, MAE_val, MSE_val, RMSE_val, MAXE_val, RMSErel_val, textstr, range_val, nb_atom, title, inter, min_ref
        pf.close_graph()
    del g
del f
del energy, force, fig_apath

test_json["is_plotted"] = True
cf.json_dump(test_json,(control_apath/("test_"+current_iteration_zfill+".json")),True)
logging.info("The DP-Test: plot phase is a success!")

del dpi, sizemult, size, ratio, figsize_ratio, colors, linestyles, props

### Cleaning
del config_json, training_iterative_apath, current_apath, control_apath
del test_json
del current_iteration, current_iteration_zfill
del deepmd_iterative_apath

del sys, Path, logging, cf, pf
del np, gc, AutoMinorLocator, plt, matplotlib
del MAE, MSE, MAXE
import gc; gc.collect(); del gc
exit()