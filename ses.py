#%% import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import HDFStorage
#%% load data
hdf = HDFStorage(r'data1.hdf5')
data = hdf.read(np.arange(0, 5000, 10, dtype = int))
#%% prepare data
data['z'], data['r'] = np.meshgrid(data['z'], data['r'])
#%% show field
def pltfield(x, y, z, u = None, v = None, dn = [1, 1], clim = [0, 1], cmap = 'viridis', clabel = '', 
    scale = 1, filename = None, figsize = (8, 6), dpi = 150, pivot = 'mid', scale_units = 'xy', units = 'xy', arrowcolor = 'w'):
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    h = ax.pcolormesh(x, y, z, cmap = cmap, clim = clim)
    if not u is None and not v is None:
        ax.quiver(x[::dn[0], ::dn[1]], y[::dn[0], ::dn[1]], u[::dn[0], ::dn[1]], v[::dn[0], ::dn[1]], 
            scale = scale, pivot = pivot, scale_units = scale_units, units = units, color = arrowcolor)
    ax.set_aspect('equal')
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    c = fig.colorbar(h, ax = ax, label = clabel)
    if not filename is None:
        fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0)

i = 1

pltfield(data['r'], data['z'], data['rho'][i], u = data['u'][i], v = data['v'][i],
    cmap = 'Purples', arrowcolor = '0',clim = [0, 40], clabel = r'$\rho$', 
    dn = [25, 25], scale = 0.2, filename = 'test-rho.png')
#%% show animation
def anifield(x, y, z, u = None, v = None, dn = [1, 1], clim = [0, 1], cmap = 'viridis', clabel = '', 
    scale = 1, filename = None, figsize = (8, 6), dpi = 150, pivot = 'mid', 
    scale_units = 'xy', units = 'xy', arrowcolor = 'w', interval = 50):

    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    temp = dict(flag = True)
    def update(i):
        h = ax.pcolormesh(x, y, z[i,:,:], cmap = cmap, clim = clim)
        if not u is None and not v is None:
            ax.quiver(x[::dn[0], ::dn[1]], y[::dn[0], ::dn[1]], u[i, ::dn[0], ::dn[1]], v[i, ::dn[0], ::dn[1]], 
                scale = scale, pivot = pivot, scale_units = scale_units, units = units, color = arrowcolor)
        ax.set_aspect('equal')
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        global flag
        if temp['flag']:
            fig.colorbar(h, ax = ax, label = clabel)
            temp['flag'] = False

    ani = FuncAnimation(fig, update, frames = range(z.shape[0]), interval = interval)
    ani.save(filename = filename, writer = 'pillow', savefig_kwargs=dict(bbox_inches = 'tight', pad_inches = 0))

anifield(data['r'], data['z'], data['rho'], u = data['u'], v = data['v'],
    cmap = 'Purples', arrowcolor = '0',clim = [0, 40], clabel = r'$\rho$', 
    dn = [25, 25], scale = 0.2, filename = 'test-rho.gif')