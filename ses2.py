#%% import
import shutil, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.ndimage as ndi
from utils import HDFStorage
#%% load data
hdf = HDFStorage('data\\data-g4f3-n3e4.hdf5')
data = hdf.read()
data['ent'] = np.log((data['g'] - 1.) * data['eint'] * np.power(data['rho'], 1. - data['g']))
#%% clear plots
path = 'data\\plots-g13f10-n3e4-dist-1d-3'
try:
    os.makedirs(path)
except:
    pass
folders = ['u', 'v', 'vmag', 'rho', 'eint', 'ent']
shutil.rmtree(path)
os.makedirs(path)
[os.makedirs(os.path.join(path, f)) for f in folders]
#%% define plot function
title = lambda r: r'r={:.1f}; $\gamma$={:.1f}'.format(r, data['g'])

def plot(t, z, r, f, ta = [0, 8, 16, 32, 64], ra = [0, 0.2], xlim = [1.1, 4.5], 
    xlabel = 'z', ylabel = '', folder = None, figsize = (8, 6), dpi = 150, 
    title = lambda s: '', extension = '.png'):

    dr = np.abs(r[1,0]-r[0,0])
    nodet = lambda t0: np.argmin(np.abs(t-t0))
    noder = lambda r0: np.abs((r-r0))<=dr/2

    tn = list(map(nodet, ta))
    rn = list(map(noder, ra))

    for i, ri in enumerate(rn):
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        [ax.plot(z[ri], f[ti][ri]) for ti in tn]
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend(labels = [str(e) for e in ta])
        ax.get_legend().set_title('t')
        ax.set_box_aspect(1)
        ax.set_title(title(ra[i]))    
        if not folder is None:
            filename = os.path.join(folder, f'{i}'+extension)
            fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
#%% test
ta = [3.5, 5.3, 7.1, 8.6, 9.3]
ra = [0, 0.2, 0.3]

# plot density
plot(data['t'], data['Z'], data['R'], data['rho'], ta = ta, ra = ra, ylabel = r'$\rho$', 
     title = title, folder = os.path.join(path, 'rho'))
#%% swept plots
# ta = [2.3, 3.5, 4.6, 5.8, 6.9, 8.1, 9.3, 10.4, 11.6, 12.8, 13.9, 15.1]
# ta = [9.3, 10.4, 11.6, 12.8, 13.9, 15.1]
ta = [3.5, 5.3, 7.1, 8.6, 9.3]
ra = [0, 0.2]

# plot density
plot(data['t'], data['Z'], data['R'], data['rho'], ta = ta, ra = ra, ylabel = r'$\rho$', 
     title = title, folder = os.path.join(path, 'rho'))

# plot energy
plot(data['t'], data['Z'], data['R'], data['eint'], ta = ta, ra = ra, ylabel = r'$Ei$', 
     title = title, folder = os.path.join(path, 'eint'))

# plot entropy
plot(data['t'], data['Z'], data['R'], data['ent'], ta = ta, ra = ra, ylabel = r'$S/c_V$', 
     title = title, folder = os.path.join(path, 'ent'))

# plot u-component velocity
plot(data['t'], data['Z'], data['R'], data['u'], ta = ta, ra = ra, ylabel = r'$u$', 
     title = title, folder = os.path.join(path, 'u'))

# plot v-component velocity
plot(data['t'], data['Z'], data['R'], data['v'], ta = ta, ra = ra, ylabel = r'$v$', 
     title = title, folder = os.path.join(path, 'v'))

# plot velocity magnitude
plot(data['t'], data['Z'], data['R'], np.hypot(data['u'], data['v']), ta = ta, ra = ra, ylabel = r'$V_{m}$', 
     title = title, folder = os.path.join(path, 'vmag'))
#%% create archive
shutil.make_archive(path, 'zip', path)