#%% import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import HDFStorage
#%% load data
hdf = HDFStorage('data-1.hdf5')
data = hdf.read(np.arange(0, 5883, 10, dtype = int))
#%% define plot
def pltfield(x, y, z, u = None, v = None, dn = [1, 1], clim = [0, 1], cmap = 'viridis', clabel = '', 
    scale = 1, filename = None, figsize = (8, 6), dpi = 150, pivot = 'mid', scale_units = 'xy', 
    units = 'xy', arrowcolor = 'w', title = ''):
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    h = ax.pcolormesh(x, y, z, cmap = cmap, clim = clim)
    if not u is None and not v is None:
        ax.quiver(x[::dn[0], ::dn[1]], y[::dn[0], ::dn[1]], u[::dn[0], ::dn[1]], v[::dn[0], ::dn[1]], 
            scale = scale, pivot = pivot, scale_units = scale_units, units = units, color = arrowcolor)
    ax.set_aspect('equal')
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    ax.set_title(title)
    c = fig.colorbar(h, ax = ax, label = clabel)
    if not filename is None:
        fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
#%% show dencity
for i in np.arange(data['rho'].shape[0]):
    pltfield(data['r'], data['z'], data['rho'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'Purples', arrowcolor = '0',clim = [0, 40], clabel = r'$\rho$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\rho\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% show energy
for i in np.arange(data['eint'].shape[0]):
    pltfield(data['r'], data['z'], data['eint'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'Purples', arrowcolor = '0',clim = [0, 1.5], clabel = r'$U$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\e\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% show u-component velocity
for i in np.arange(data['u'].shape[0]):
    pltfield(data['r'], data['z'], data['u'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', arrowcolor = '0',clim = [-0.05, 0.05], clabel = r'$u$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\u\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% show u-component velocity
for i in np.arange(data['v'].shape[0]):
    pltfield(data['r'], data['z'], data['v'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', arrowcolor = '0',clim = [-0.05, 0.05], clabel = r'$v$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\v\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% show velocity magnitude
for i in np.arange(data['v'].shape[0]):
    pltfield(data['r'], data['z'], np.hypot(data['u'][i], data['v'][i]), u = data['u'][i], v = data['v'][i],
        cmap = 'Grays', arrowcolor = '0',clim = [0, 0.15], clabel = r'$Vm$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\vm\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% process vorticity
import scipy.ndimage as ndi
sobker = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
data['dudr'] = ndi.convolve(data['u'], sobker, mode = 'reflect', axes=(1,2))/data['dr']
data['dudz'] = ndi.convolve(data['u'], sobker.T, mode = 'reflect', axes=(1,2))/data['dz']
data['dvdr'] = ndi.convolve(data['v'], sobker, mode = 'reflect', axes=(1,2))/data['dr']
data['dvdz'] = ndi.convolve(data['v'], sobker.T, mode = 'reflect', axes=(1,2))/data['dz']
data['rot'] = data['dudz'] - data['dvdr']
#%% show vorticity
for i in np.arange(data['rho'].shape[0]):
    pltfield(data['r'], data['z'], data['rot'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', arrowcolor = '0',clim = [-2, 2], clabel = r'$\omega$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\rot\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))
#%% process Q-criteria
def procqcrit(dudx, dudy, dvdx, dvdy):
    gradvel= np.stack((dudx, dudy, dvdx, dvdy)).reshape((2,2,)+dudx.shape)
    permute = (1,0,)+tuple(np.arange(gradvel.ndim)[2:])
    symm = 1/2*(gradvel + np.transpose(gradvel, axes = permute))
    skew = 1/2*(gradvel - np.transpose(gradvel, axes = permute))
    det2d = lambda x: x[0,0,:]*x[1,1,:]-x[1,0,:]*x[0,1,:]
    q = det2d(skew)**2 - det2d(symm)**2
    q = np.where(q>0, q, 0)
    return q

data['q'] = procqcrit(data['dudr'], data['dudz'], data['dvdr'], data['dvdz'])
#%% show Q-criteria
for i in np.arange(data['rho'].shape[0]):
    pltfield(data['r'], data['z'], data['q'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'Grays', arrowcolor = '0',clim = [0, 5], clabel = r'$Q_{crit.}$', 
        dn = [25, 25], scale = 0.2, filename = f'data\\plots\\q\\{i}.png', 
        title = 't={:.3f}'.format(data['t'][i]))