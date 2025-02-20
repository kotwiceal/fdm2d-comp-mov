#%% import
import shutil, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.ndimage as ndi
from utils import HDFStorage
#%% load data
hdf = HDFStorage('data\\data-g5f3-n3e4.hdf5')
data = hdf.read()
data['range'] = np.arange(0,data['u'].shape[0],5)
# process entrophy
data['ent'] = np.log((data['g'] - 1.) * data['eint'] * np.power(data['rho'], 1. - data['g']))
#%% process vorticity
sobker = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
data['dudr'] = ndi.convolve(data['u'], sobker, mode = 'reflect', axes=(1,2))/data['dr']
data['dudz'] = ndi.convolve(data['u'], sobker.T, mode = 'reflect', axes=(1,2))/data['dz']
data['dvdr'] = ndi.convolve(data['v'], sobker, mode = 'reflect', axes=(1,2))/data['dr']
data['dvdz'] = ndi.convolve(data['v'], sobker.T, mode = 'reflect', axes=(1,2))/data['dz']
data['rot'] = data['dudz'] - data['dvdr']
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
#%% clear plots
path = 'data\\plots-g5f3-n3e4-crop2'
folders = ['u', 'v', 'vmag', 'rho', 'eint', 'rot', 'q', 'ent']
shutil.rmtree(path)
os.makedirs(path)
[os.makedirs(os.path.join(path, f)) for f in folders]
#%% define plot
def pltfield(x, y, z, u = None, v = None, dn = [10, 10], clim = [0, 1], cmap = 'viridis', clabel = '', 
    scale = 0.5, filename = None, figsize = (8, 6), dpi = 150, pivot = 'mid', scale_units = 'xy', 
    units = 'xy', arrowcolor = '0', title = '', norm = colors.Normalize, xlim = [0, 1], ylim = [1.5, 2.4]):

    norm = norm(vmin = clim[0], vmax = clim[1])
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    pcm = ax.pcolormesh(x, y, z, cmap = cmap, norm = norm) 
    if not u is None and not v is None:
        q = ax.quiver(x[::dn[0], ::dn[1]], y[::dn[0], ::dn[1]], u[::dn[0], ::dn[1]], v[::dn[0], ::dn[1]], 
            scale = scale, pivot = pivot, scale_units = scale_units, units = units, color = arrowcolor)
    ax.set_aspect('equal')
    ax.set_xlabel('r')
    ax.set_ylabel('z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    fig.colorbar(pcm, ax = ax, label = clabel)
    if not filename is None:
        fig.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
titlehandle = lambda i: r't={:.1f} \n $\gamma$={:.1f}'.format(data['t'][i], data['g'])
#%% swept plots
for i in data['range']:
    # plot density
    pltfield(data['R'], data['Z'], data['rho'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.jet_r, clim = [data['rho'].min(), data['rho'].max()], clabel = r'$\rho$', 
        filename = os.path.join(path, f'rho\\{i}.png'), 
        title = titlehandle(i), norm = colors.LogNorm)
    
    # plot energy
    pltfield(data['R'], data['Z'], data['eint'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.hot_r, clim = [data['eint'].min(), data['eint'].max()], clabel = r'$Ei$', 
        filename = os.path.join(path, f'eint\\{i}.png'), 
        title = titlehandle(i), norm = colors.LogNorm)
    
    # plot entropy
    pltfield(data['R'], data['Z'], data['ent'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.cool_r, clim = [data['ent'].min(), data['ent'].max()], clabel = r'$S/c_V$', 
        filename = os.path.join(path, f'ent\\{i}.png'), 
        title = titlehandle(i))
    
    # plot u-component velocity
    pltfield(data['R'], data['Z'], data['u'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.1, 0.1], clabel = r'$u$', 
        filename = os.path.join(path, f'u\\{i}.png'), 
        title = titlehandle(i))

    # plot v-component velocity
    pltfield(data['R'], data['Z'], data['v'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.1, 0.1], clabel = r'$v$', 
        filename = os.path.join(path, f'v\\{i}.png'), 
        title = titlehandle(i))

    # plot velocity magnitude
    pltfield(data['R'], data['Z'], np.hypot(data['u'][i], data['v'][i]), u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.3, 0.3], clabel = r'$V_{m}$', 
        filename = os.path.join(path, f'vmag\\{i}.png'), 
        title = titlehandle(i))

    # plot vorticity
    pltfield(data['R'], data['Z'], data['rot'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-3, 3], clabel = r'$\omega$', 
        filename = os.path.join(path, f'rot\\{i}.png'), 
        title = titlehandle(i))
    
    # plot Q-criteria
    pltfield(data['R'], data['Z'], data['q'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'Blues', clim = [0, 10], clabel = r'$Q_{crit.}$', 
        filename = os.path.join(path, f'q\\{i}.png'), 
        title = titlehandle(i))
#%% create archive
shutil.make_archive(path, 'zip', path)