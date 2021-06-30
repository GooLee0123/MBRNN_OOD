import os
import logging

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class Dataset(Dataset):
    """
        Load dataset
    """

    def __init__(self, phase, opt, dopt='ID'):
        self.logger = logging.getLogger(__name__)

        self.dopt = dopt
        self.phase = phase

        self.ind = opt.ind
        self.ncls = opt.ncls
        self.train = opt.train
        self.method = opt.method
        self.bin_dn = opt.bin_dn
        self.data_dn = opt.data_dn
        self.ul_prefix = opt.ul_prefix
        self.tr_ul_prefix = opt.tr_ul_prefix

        if self.dopt == 'UL':
            self.X, self.y = self._load_data()
        else:
            self.X, self.y, self.z, self.zcls, self.binc = self._load_data()

        self.len = self.X.size()[0]

    def unnorm(self, X):
        ref_dn = './data_original'
        ref_fn = os.path.join(ref_dn, 'PS-DR1_galaxy.txt')
        refX = pd.read_csv(ref_fn, header=None, delimiter=' ').values
        refX = refX[:, 4:21].astype(np.float32)
        refX[:, -1] = np.log(refX[:, -1])
        refX_min, refX_max = refX.min(0), refX.max(0)

        unnormed = X*(refX_max-refX_min) + refX_min
        unnormed[:, -1] = np.exp(unnormed[:, -1])

        return unnormed

    def zbinning(self, z, bins=None):
        _z = z.ravel()

        if self.dopt == 'LOOD' and self.method == 'unsup':
            zcls, binc = (None, None)
        else:
            bin_fn = '%s/%s_redshifts_%s-uniform.txt' % \
                (self.bin_dn, self.ind, self.ncls)
            log_msg = "Read uniform bins for %s from %s" % (self.phase, bin_fn)
            self.logger.info(log_msg)

            bins = np.genfromtxt(bin_fn)
            zcls = torch.from_numpy(np.digitize(_z, bins)-1)
            zcls = torch.where(zcls < 0, torch.tensor(0), zcls)
            zcls = torch.where(
                zcls >= self.ncls, torch.tensor(self.ncls-1), zcls)
            binc = torch.from_numpy(
                np.array([(lb+ub)/2. for lb, ub in zip(bins[:-1], bins[1:])],
                         dtype=np.float32))

        return zcls, binc

    def _zfilter(self, z):
        z[z < 0] = 0

        return z

    def _load_data(self):
        if self.dopt == 'ID':
            _prefix = self.ind
        elif self.dopt == 'UL':
            _prefix = 'unlabeled'
        else:
            _prefix = 'labeled_ood'
        fn = '%s/PS-DR1_%s_%s.npy' % (self.data_dn, _prefix, self.phase)
        if self.dopt == 'UL':
            prf = self.tr_ul_prefix if self.train else self.ul_prefix
            fn = fn.replace('.npy', '_%s.npy' % prf)
        self.logger.info("load data from %s" % fn)

        # data = np.load(fn)
        p = Path(fn)
        with p.open('rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            data = np.load(f)
            while f.tell() < fsz:
                data = np.vstack((data, np.load(f)))
        y = np.array(data[:, -1], dtype=np.long)

        if self.dopt == 'UL':
            x = np.array(data[:, :-1], dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            x = torch.from_numpy(np.array(data[:, 1:-1], dtype=np.float32))
            z = np.array(data[:, 0].reshape(-1, 1), dtype=np.float32)
            z = self._zfilter(z)
            zcls, binc = self.zbinning(z)
            z = torch.from_numpy(np.array(z, dtype=np.float32))

            return x, y, z, zcls, binc

    def update_len(self, new_len):
        self.len = new_len

    def __getitem__(self, index):
        if self.dopt == 'UL' or self.dopt == 'LOOD':
            X, y = self.X[index], self.y[index]
            # if self.method == 'unsup':
            return X, y
            # else:
            #     zcls = self.zcls[index]
            #     return X, y, zcls
        else:
            X, zcls = self.X[index], self.zcls[index]
            return X, zcls

    def __len__(self):
        return self.len
