import shutil, os
import datetime, h5py
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

class HDFStorage:
    """HDF files manager for data accumulation at long-time processing."""
    def __init__(self, path: str, dtype: str = 'f8', decimate: int = 32):
        if path == None:
            path = datetime.datetime.now().isoformat() + '.hdf5'
        self.path = path

        self.dtype = dtype
        self.decimate = decimate

        self.cntdec = 0

    def write(self, **kwargs):
        """Store passing data in the dataset."""
        with h5py.File(self.path, 'a') as fid:
            for key, value in kwargs.items():
                if key in fid:
                    del fid[key]
                fid.create_dataset(key, value.shape, data = value)

    def append(self, *args, **kwargs):
        """Append passing data in the group."""
        with h5py.File(self.path, 'a') as fid:
            for key, value in kwargs.items():
                if key not in fid.keys():
                    fid.create_group(key)
                if args is tuple():
                    index = str(len(fid[key]) + 1)
                else:
                    index = str(args[0])
                fid[key].create_dataset(index, data = value, 
                    dtype = self.dtype)

    def read(self, index: list = None, step: int = None):
        """Read all data."""
        data = dict()
        with h5py.File(self.path, 'r') as fid:
            for key, value in fid.items():
                if isinstance(fid[key], h5py.Dataset):
                    data[key] = value[()]
                else:
                    n = len(fid[key])
                    if index is None and step is None:
                        index = np.arange(n)
                    else:
                        if index is None:
                            index = np.arange(0, n, step)
                        else:
                            index= np.array(index)
                    data[key] = np.zeros((index.shape[0],) + value['1'].shape)
                    for i, v in enumerate(index):
                        data[key][i] = value[str(v+1)][()]
        return data

    def watch(self, *args, **kwargs):
        """Append passing data with specified decimation."""
        if (self.cntdec == self.decimate):
            self.cntdec = 0
            self.append(*args, **kwargs)
        else:
            self.cntdec = self.cntdec + 1

def procvelgrad(u, v, dx = None, dy = None):
    # process velocity gradient and vorticity
    if dx is None: dx = 1
    if dy is None: dy = 1
    sobker = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    dudx = ndi.convolve(u, sobker, mode = 'reflect', axes=(1,2))/dx
    dudy = ndi.convolve(u, sobker.T, mode = 'reflect', axes=(1,2))/dx
    dvdx = ndi.convolve(v, sobker, mode = 'reflect', axes=(1,2))/dy
    dvdy = ndi.convolve(v, sobker.T, mode = 'reflect', axes=(1,2))/dy
    rot = dudx - dvdy
    return dudx, dudy, dvdx, dvdy, rot

def procqcrit(dudx, dudy, dvdx, dvdy):
    # process Q-criteria
    gradvel= np.stack((dudx, dudy, dvdx, dvdy)).reshape((2,2,)+dudx.shape)
    permute = (1,0,)+tuple(np.arange(gradvel.ndim)[2:])
    symm = 1/2*(gradvel + np.transpose(gradvel, axes = permute))
    skew = 1/2*(gradvel - np.transpose(gradvel, axes = permute))
    det2d = lambda x: x[0,0,:]*x[1,1,:]-x[1,0,:]*x[0,1,:]
    q = det2d(skew)**2 - det2d(symm)**2
    q = np.where(q>0, q, 0)
    return q

def plotmkdir(path):
    try:
        os.makedirs(path)
    except:
        pass
    folders = ['u', 'v', 'vmag', 'rho', 'eint', 'rot', 'q', 'ent', 'M']
    shutil.rmtree(path)
    os.makedirs(path)
    [os.makedirs(os.path.join(path, f)) for f in folders]

def plot2d(x, y, z, u = None, v = None, dn = [10, 10], clim = [0, 1], cmap = 'viridis', clabel = '', 
    scale = 0.5, filename = None, figsize = (8, 6), dpi = 150, pivot = 'mid', scale_units = 'xy', 
    units = 'xy', arrowcolor = '0', title = '', norm = colors.Normalize, 
    xlim = [0, 1], ylim = [1.5, 4], type = 'polymesh', levels = 50):
    # show 2D plot

    norm = norm(vmin = clim[0], vmax = clim[1])
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    match type:
        case 'polymesh':
            pcm = ax.pcolormesh(x, y, z, cmap = cmap, norm = norm) 
        case 'contourf':
            pcm = ax.contourf(x, y, z, levels, cmap = cmap, norm = norm) 
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