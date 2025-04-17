#%% import
import shutil, os
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from utils import HDFStorage, procqcrit, procvelgrad, plotmkdir, plot2d
#%% load data
hdf = HDFStorage('data\\data-g1.3-n3e4-phi0.05.hdf5')
data = hdf.read(index=[0,1,2])
data['range'] = np.arange(0,data['u'].shape[0],5)
#%% process entrophy
data['ent'] = np.log((data['g'] - 1.) * data['eint'] * np.power(data['rho'], 1. - data['g']))
#%% process vorticity
data['dudr'], data['dudz'], data['dvdr'], data['dvdz'], data['rot'] = procvelgrad(data['u'], data['v'], dx = data['dr'], dy = data['dz'])
#%% process Q-criteria
data['q'] = procqcrit(data['dudr'], data['dudz'], data['dvdr'], data['dvdz'])
#%% create plot storage folders & clear plots
data['path'] = 'data\\plots-g1.3-n3e4-phi0.05'
plotmkdir(data['path'])
#%% define title handle
titlehandle = lambda i: r't={:.1f}; $\gamma$={:.1f}'.format(data['t'][i], data['g'])
#%% swept plots
for i in data['range']:
    # plot density
    plot2d(data['R'], data['Z'], data['rho'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.jet_r, clim = [data['rho'].min(), data['rho'].max()], clabel = r'$\rho$', 
        filename = os.path.join(data['path'], 'rho', f'{i}.png'), 
        title = titlehandle(i), norm = colors.LogNorm)
    
    # plot energy
    plot2d(data['R'], data['Z'], data['eint'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.hot_r, clim = [data['eint'].min(), data['eint'].max()], clabel = r'$Ei$', 
        filename = os.path.join(data['path'], 'eint', f'{i}.png'), 
        title = titlehandle(i), norm = colors.LogNorm)
    
    # plot entropy
    plot2d(data['R'], data['Z'], data['ent'][i], u = data['u'][i], v = data['v'][i],
        cmap = cm.cool_r, clim = [data['ent'].min(), data['ent'].max()], clabel = r'$S/c_V$', 
        filename = os.path.join(data['path'], 'ent', f'{i}.png'), 
        title = titlehandle(i))
    
    # plot u-component velocity
    plot2d(data['R'], data['Z'], data['u'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.1, 0.1], clabel = r'$u$', 
        filename = os.path.join(data['path'], 'u', f'{i}.png'), 
        title = titlehandle(i))

    # plot v-component velocity
    plot2d(data['R'], data['Z'], data['v'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.1, 0.1], clabel = r'$v$', 
        filename = os.path.join(data['path'], 'v', f'{i}.png'), 
        title = titlehandle(i))

    # plot velocity magnitude
    plot2d(data['R'], data['Z'], np.hypot(data['u'][i], data['v'][i]), u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-0.3, 0.3], clabel = r'$V_{m}$', 
        filename = os.path.join(data['path'], 'vmag', f'{i}.png'), 
        title = titlehandle(i))

    # plot vorticity
    plot2d(data['R'], data['Z'], data['rot'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'bwr', clim = [-3, 3], clabel = r'$\omega$', 
        filename = os.path.join(data['path'], 'rot', f'{i}.png'), 
        title = titlehandle(i))
    
    # plot Q-criteria
    plot2d(data['R'], data['Z'], data['q'][i], u = data['u'][i], v = data['v'][i],
        cmap = 'Blues', clim = [0, 10], clabel = r'$Q_{crit.}$', 
        filename = os.path.join(data['path'], 'q', f'{i}.png'), 
        title = titlehandle(i))
#%% create archive
shutil.make_archive(data['path'], 'zip', data['path'])