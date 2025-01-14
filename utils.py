import datetime, h5py
import numpy as np

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
                fid[key].create_dataset(index, value.shape, data = value, 
                    dtype = self.dtype)

    def read(self, index: list = None):
        """Read all data."""
        data = dict()
        with h5py.File(self.path, 'r') as fid:
            for key, value in fid.items():
                if isinstance(fid[key], h5py.Dataset):
                    data[key] = value[:]
                else:
                    if index is None:
                        n = len(fid[key])
                        index = np.arange(n)
                    else:
                        n = index.size
                    data[key] = np.zeros((n,) + value['1'].shape)
                    for i, v in enumerate(index):
                        data[key][i] = value[str(v+1)][:]
        return data

    def watch(self, *args, **kwargs):
        """Append passing data with specified decimation."""
        if (self.cntdec == self.decimate):
            self.cntdec = 0
            self.append(*args, **kwargs)
        else:
            self.cntdec = self.cntdec + 1