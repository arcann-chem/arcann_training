import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import gc

def anint(a):
    return np.sign(a)*np.ceil(np.abs(a))

def max_range(IN_val,IN_div=5):
    f_val=np.abs(IN_val)
    if f_val > IN_div:
        return int(np.ceil(f_val/IN_div)*IN_div)
    else:
        return IN_div

def axes(ax,sizemult):
    plt.setp(ax.spines.values(),linewidth=sizemult*4)
    ax.axes.tick_params(which='minor',size=sizemult*6,width=sizemult*2)
    ax.axes.tick_params(which='major',size=sizemult*8,width=sizemult*4)
    ax.xaxis.labelpad=sizemult*15
    ax.yaxis.labelpad=sizemult*15

def xaxe(ax,xmin,xmax,div,div_min,sformat,sizemult):
    ax.set_xlim(xmin,xmax)
    ax.axes.set_xticks(np.arange(xmin, xmax+div, div))
    ax.axes.set_xticklabels([sformat.format(x) for x in np.arange(xmin, xmax+div, div)],fontsize=sizemult*16,fontweight='bold')
    if div_min != 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=div_min))

def xaxe_auto(ax,div_min,sformat,sizemult):
    ax.axes.set_xticks(ax.axes.get_xticks())
    ax.axes.set_xticklabels([sformat.format(x) for x in ax.axes.get_xticks()],fontsize=sizemult*16,fontweight='bold')
    if div_min != 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=div_min))

def yaxe(ax,ymin,ymax,div,div_min,sformat,sizemult):
    ax.set_ylim(ymin,ymax)
    ax.axes.set_yticks(np.arange(ymin, ymax+div, div))
    ax.axes.set_yticklabels([sformat.format(x) for x in np.arange(ymin, ymax+div, div)],fontsize=sizemult*16,fontweight='bold')
    if div_min != 0:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=div_min))

def yaxe_auto(ax,div_min,sformat,sizemult):
    ax.axes.set_yticks(ax.axes.get_yticks())
    ax.axes.set_yticklabels([sformat.format(x) for x in ax.axes.get_yticks()],fontsize=sizemult*16,fontweight='bold')
    if div_min != 0:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=div_min))

def close_graph():
    plt.cla();plt.clf();plt.close()
    plt.close('all')
    gc.collect()