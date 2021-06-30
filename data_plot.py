import os
import logging
import argparse

import numpy as np

import plotter
from utils import str2bool


LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)

parser = argparse.ArgumentParser()
parser.add_argument('--InD',
                    default='galaxy',
                    type=str, dest='InD')
parser.add_argument('--data_prefix',
                    default='RA_10_100',
                    type=str, dest='data_prefix')
parser.add_argument('--kind',
                    default='scatter',
                    type=str, dest='kind')


opt = parser.parse_args()

outdn = './DataPlot'+'_ind_'+opt.InD
# subdn = 'ID' if not opt.data_prefix else opt.data_prefix.upper()
# outdn = os.path.join(outdn, subdn)


def get_fn():
    datadn = './data_processed_ind_'+opt.InD
    dphases = ['train', 'val', 'eval']
    dfns = []
    for dphase in dphases:
        if not opt.data_prefix:
            tempfn = os.path.join(datadn, 'PS-DR1_galaxy_%s.npy' % dphase)
        else:
            tempfn = 'PS-DR1_unlabeled_%s_%s.npy' % (dphase, opt.data_prefix)
            tempfn = os.path.join(datadn, tempfn)
        dfns.append(tempfn)

    return dfns


def read_data(fns):
    colors = []
    for fn in fns:
        logging.info("Load data from %s" % fn)
        data = np.load(fn)
        if not opt.data_prefix:
            color = data[:, 1:17]
        else:
            color = data[:, :-1]

        colors.append(color)

    return colors


def main():
    fns = get_fn()
    colors = read_data(fns)

    # plotter.DatasetColorNumberDensity(*colors,
    #                                   dtype=opt.data_prefix.upper(),
    #                                   ID=opt.InD)
    plotter.DatasetColorPairPlot(np.vstack(colors)[:, ::2],
                                 dtype=opt.data_prefix.upper(),
                                 ID=opt.InD, kind=opt.kind)
    plotter.DatasetColorPairPlot(np.vstack(colors)[:, 1::2],
                                 dtype=opt.data_prefix.upper(),
                                 ID=opt.InD, err=True, kind=opt.kind)


if __name__ == '__main__':
    main()