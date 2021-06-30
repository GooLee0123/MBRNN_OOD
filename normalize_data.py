import os
import logging
import argparse

import pandas as pd
import numpy as np

from pathlib import Path
from option_parse import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--data_plot',
                    default=False, type=str2bool,
                    dest='data_plot')
parser.add_argument('--indn',
                    default='./data_original',
                    type=str, dest='indn')
parser.add_argument('--ID',
                    default='PS-DR1_galaxy.txt',
                    type=str, dest='ID')
parser.add_argument('--labeled_OOD',
                    default='use-cleaned-entire_sample_PS-DR1_data_with_EB-V.txt.qso,use-cleaned-entire_sample_PS-DR1_data_with_EB-V.txt.star',
                    type=str, dest='LOOD')
parser.add_argument('--unlabeled',
                    default='cleaned-PS-DR1_data_with_EB-V-RA_ge_100_and_lt_110.txt',
                    type=str, dest='UL')
parser.add_argument('--save_ul_only',
                    default=False,
                    type=str2bool, dest='save_ul_only')
parser.add_argument('--save_lo_only',
                    default=True,
                    type=str2bool, dest='save_lo_only')
parser.add_argument('--ul_usample',
                    default=False,
                    type=str2bool, dest='ul_usample')

opt = parser.parse_args()

if ',' in opt.UL:
    opt.ul_prefix = 'RA_combined'
else:
    RAsplit = opt.UL.split('_')
    RArange = tuple([RAsplit[-4], RAsplit[-1][:-4]])
    opt.ul_prefix = 'RA_%s_%s' % RArange
if opt.ul_usample:
    opt.ul_prefix += '_usample'

flog = 'logs'
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    filemode='w',
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)

logging.info(opt)

schunk = 1000000

ind = opt.ID.split('_')[-1].replace('.txt', '')

outdn = './data_processed_test'
outdn += '_ind_'+ind

if not os.path.exists(outdn):
    os.makedirs(outdn)

out_prefixs = [ind, 'labeled_ood', 'unlabeled']

rseed = 321
unsample = 1000000
np.random.seed(rseed)


def get_data_fn():
    dpath = opt.indn
    id_fns = opt.ID.split(',')
    lood_fns = opt.LOOD.split(',')
    ul_fns = opt.UL.split(',')

    IDfns = [os.path.join(dpath, fn) for fn in id_fns]
    LOODfns = [os.path.join(dpath, fn) for fn in lood_fns]
    ULfns = [os.path.join(dpath, fn) for fn in ul_fns]
    PSCfns = [ULfn.replace('-DR1_data_with_EB-V', 'C') for ULfn in ULfns]

    return [IDfns, LOODfns, ULfns], PSCfns


def data_process(raw, labels):
    def _process(inputs, i, label, minmax=None):
        if i != 2:
            ids = inputs[:, 0:1]
            zspec = inputs[:, 2:3].astype(np.float32)  # [n, 1]
            colors = inputs[:, 4:21].astype(np.float32)  # [n, 17]

            colors[:, -1] = np.log(colors[:, -1])

            if minmax is None:
                logging.info("get min-max values")
                input_min = np.min(colors, 0)
                input_max = np.max(colors, 0)
                minmax = (input_min, input_max)
            else:
                logging.info("use pre-computed min-max values")
                input_min, input_max = minmax

            color_normed = (colors-input_min)/(input_max-input_min)
            processed = np.hstack((zspec, color_normed, label))
        else:
            ids = []
            processed = []
            for inp in inputs:
                temp_id = inp[:, 17:18]
                colors = inp[:, :17].astype(np.float32)

                colors[:, -1] = np.log(colors[:, -1])
                label = np.zeros(temp_id.shape) + i

                logging.info("use pre-computed min-max values")
                input_min, input_max = minmax

                color_normed = (colors-input_min)/(input_max-input_min)

                ids.append(temp_id)
                processed.append(np.hstack((color_normed, label)))
            ids = np.vstack(ids)
            processed = np.vstack(processed)

        return ids, processed, minmax

    minmax = None
    id_container = []
    processed_container = []
    # raw data: [[data_id], [data_lood], [data_ul]]
    for i, data in enumerate(raw):
        if i != 2:
            data = np.vstack(data)
            label = np.hstack(labels[i]).reshape(-1, 1)
        ids, processed, minmax = _process(data, i, label, minmax)

        id_container.append(ids)  # [nparr, nparr, list]
        processed_container.append(processed)  # [nparr, nparr, list]

    return id_container, processed_container


def get_data(dtypes):
    data = [[], [], []]
    labels = [[], [], []]
    for i, fns in enumerate(dtypes):
        for fn in fns:
            logging.info("Load data from %s" % fn)
            temp_data = pd.read_csv(fn, header=None,
                                    delimiter=' ',
                                    chunksize=schunk,
                                    encoding='cp1252')
            for j, tdata in enumerate(temp_data):
                if 'RA_combined' not in opt.ul_prefix and j == 0 and i == 2:
                    continue
                td = tdata.values
                data[i].append(td)
                if i != 1:
                    add = i
                else:
                    add = i if 'qso' in fn else i+2  # qso 1, star 3
                labels[i].append(np.zeros(len(td))+add)
                if opt.ul_usample and i == 2:
                    break

    return data, labels


def get_psc(fns):
    psc = []
    for fn in fns:
        logging.info("Load PSC from %s" % fn)
        temp_psc = pd.read_csv(fn, header=None,
                               delimiter=' ',
                               chunksize=schunk)
        for i, tpsc in enumerate(temp_psc):
            if 'RA_combined' not in opt.ul_prefix and i == 0:
                continue
            psc.append(tpsc.values)
            if opt.ul_usample:
                break

    return psc


def get_gll(fns):
    gll = []
    for fn in fns:
        logging.info("Load data from %s" % fn)
        temp_data = pd.read_csv(fn, header=None,
                                delimiter=' ',
                                chunksize=schunk)
        for tdata in temp_data:
            gll.append(tdata.values[:, -2:])
            if opt.ul_usample:
                break

    return gll


def save_data(phase, dtype, data):
    ids, processed = data
    id_fn = os.path.join(outdn, 'PS-DR1_%s_%s_ID.npy' % (phase, dtype))
    proc_fn = os.path.join(outdn, 'PS-DR1_%s_%s.npy' % (phase, dtype))
    if os.path.exists(id_fn):
        os.remove(id_fn)
    if os.path.exists(proc_fn):
        os.remove(proc_fn)
    # if isinstance(ids, list):
    #     pid = Path(id_fn)
    #     with pid.open('ab') as f:
    #         for temp_id in ids:
    #             np.save(f, temp_id)
    #             logging.info("ids with shape %s is saved at %s" %
    #                          (temp_id.shape, id_fn))

    #     pdat = Path(proc_fn)
    #     with pdat.open('ab') as f:
    #         for temp_proc in processed:
    #             np.save(f, temp_proc)
    #             logging.info("data with shape %s is saved at %s" %
    #                          (temp_proc.shape, proc_fn))
    # else:
    np.save(id_fn, ids)
    np.save(proc_fn, processed)
    logging.info("ids with shape %s is saved at %s" % (ids.shape, id_fn))
    logging.info("data with shape %s is saved at %s" %
                 (processed.shape, proc_fn))


def save_psc(dtype, psc):
    psc_fn = os.path.join(outdn, 'PSC_%s.npy' % dtype)
    if os.path.exists(psc_fn):
        os.remove(psc_fn)

    # if isinstance(psc, list):
    #     pdat = Path(psc_fn)
    #     with pdat.open('ab') as f:
    #         for temp_psc in psc:
    #             np.save(f, temp_psc)
    #             logging.info("psc with shape %s is saved at %s" %
    #                          (temp_psc.shape, psc_fn))
    # else:
    np.save(psc_fn, psc)
    logging.info("psc with shape %s is saved at %s" %
                 (psc.shape, psc_fn))


def save_gll(dtype, gll):
    gll_fn = os.path.join(outdn, 'GLL_%s.npy' % dtype)
    if os.path.exists(gll_fn):
        os.remove(gll_fn)

    # if isinstance(gll, list):
    #     pdat = Path(gll_fn)
    #     with pdat.open('ab') as f:
    #         for temp_gll in gll:
    #             np.save(f, temp_gll)
    #             logging.info("gll with shape %s is saved at %s" %
    #                          (temp_gll.shape, gll_fn))
    # else:
    np.save(gll_fn, gll)
    logging.info("gll with shape %s is saved at %s" %
                 (gll.shape, gll_fn))


def split_and_save_data(idts, raw, processed):
    IDrate = [0.8, 0.1]
    LOODrate = [0., 0.]
    ULrate = [0.8, 0.1]

    rates = [IDrate, LOODrate, ULrate]

    phases = ['train', 'val', 'eval']
    for i, (ids, proc) in enumerate(zip(idts, processed)):
        if i != 2 and not opt.save_ul_only:
            if opt.save_lo_only:
                if i == 0:
                    logging.info("Jump to LOOD")
                    continue

            dlen = len(ids)

            if i == 0:
                ridx_fn = 'id_rand_idx.txt'
            else:
                ridx_fn = 'ood_rand_idx.txt'
            ridx_dn = './rand_idx'
            ridx_fn = os.path.join(ridx_dn, ridx_fn)
            if os.path.exists(ridx_fn):
                rand_idx = np.genfromtxt(ridx_fn).astype(np.long)
            else:
                rand_idx = np.arange(0, dlen, dtype=np.long)
                np.random.shuffle(rand_idx)
                np.savetxt(ridx_fn, rand_idx)

            ids = ids[rand_idx]
            proc = proc[rand_idx]

            idx_train = int(dlen*rates[i][0])
            idx_val = idx_train + int(dlen*rates[i][1])
            idx = [0, idx_train, idx_val, dlen]

            print(idx)

            for j, phase in enumerate(phases):
                temp_ids = ids[idx[j]:idx[j+1]]
                temp_proc = proc[idx[j]:idx[j+1]]

                print(len(temp_ids))
                if len(temp_ids) != 0:
                    save_data(out_prefixs[i], phase, (temp_ids, temp_proc))
        elif i == 2 and not opt.save_lo_only:
            # dlen = np.sum([len(tid) for tid in ids])
            dlen = len(ids)

            ridx_fn = 'ul_rand_idx_%s.txt' % opt.ul_prefix
            ridx_dn = './rand_idx'
            ridx_fn = os.path.join(ridx_dn, ridx_fn)
            if os.path.exists(ridx_fn):
                rand_idx = np.genfromtxt(ridx_fn).astype(np.long)
            else:
                rand_idx = np.arange(0, dlen, dtype=np.long)
                np.random.shuffle(rand_idx)
                np.savetxt(ridx_fn, rand_idx)

            idx_train = int(dlen*rates[i][0])
            idx_val = idx_train + int(dlen*rates[i][1])
            idx = [0, idx_train, idx_val, dlen]

            ids_np = ids[rand_idx]
            proc_np = proc[rand_idx]

            raw = np.vstack(raw[-1])[rand_idx]

            for j, phase in enumerate(phases):
                temp_ids = ids_np[idx[j]:idx[j+1]]
                temp_proc = proc_np[idx[j]:idx[j+1]]

                temp_raw = raw[idx[j]:idx[j+1]]
                save_raw_data(temp_raw, phase)

                # temp_dlen = len(temp_ids)
                # if temp_dlen > schunk:
                #     nchunk = len(temp_ids) // schunk + 1
                #     temp_ids = np.array_split(temp_ids, nchunk)
                #     temp_proc = np.array_split(temp_proc, nchunk)

                save_data(out_prefixs[i],
                          phase+'_'+opt.ul_prefix, (temp_ids, temp_proc))


def split_and_save_psc(psc):
    ULrate = [0.8, 0.1]

    phases = ['train', 'val', 'eval']
    dlen = np.sum([len(tpsc) for tpsc in psc])

    ridx_fn = 'ul_rand_idx_%s.txt' % opt.ul_prefix
    ridx_dn = './rand_idx'
    ridx_fn = os.path.join(ridx_dn, ridx_fn)
    if os.path.exists(ridx_fn):
        rand_idx = np.genfromtxt(ridx_fn).astype(np.long)
    else:
        raise NameError("Can't find %s" % rand_idx)

    idx_train = int(dlen*ULrate[0])
    idx_val = idx_train + int(dlen*ULrate[1])
    idx = [0, idx_train, idx_val, dlen]

    psc_np = np.vstack(psc)[rand_idx]
    for i, phase in enumerate(phases):
        temp_psc = psc_np[idx[i]:idx[i+1]]
        # temp_dlen = len(temp_psc)
        # if temp_dlen > schunk:
        #     nchunk = len(temp_psc) // schunk + 1
        #     temp_psc = np.array_split(temp_psc, nchunk)

        save_psc(phase+'_'+opt.ul_prefix, temp_psc)


def split_and_save_gll(gll):
    ULrate = [0.8, 0.1]

    phases = ['train', 'val', 'eval']
    dlen = np.sum([len(tgll) for tgll in gll])

    ridx_fn = 'ood_rand_idx_%s.txt' % opt.ul_prefix
    ridx_dn = './rand_idx'
    ridx_fn = os.path.join(ridx_dn, ridx_fn)
    if os.path.exists(ridx_fn):
        rand_idx = np.genfromtxt(ridx_fn).astype(np.long)
    else:
        raise NameError("Can't find %s" % rand_idx)

    idx_train = int(dlen*ULrate[0])
    idx_val = idx_train + int(dlen*ULrate[1])
    idx = [0, idx_train, idx_val, dlen]

    gll_np = np.vstack(gll)[rand_idx]
    for i, phase in enumerate(phases):
        temp_gll = gll_np[idx[i]:idx[i+1]]
        # temp_dlen = len(temp_gll)
        # if temp_dlen > schunk:
        #     nchunk = len(temp_gll) // schunk + 1
        #     temp_gll = np.array_split(temp_gll, nchunk)

        save_gll(phase+'_'+opt.ul_prefix, temp_gll)


def save_raw_data(raw, key):
    fn = os.path.join(outdn, "uniformly_sampled_raw_ul_%s.npy" % key)
    np.save(fn, raw)
    logging.info("Raw data is saved at %s" % fn)


def main():
    dtypes, PSCfns = get_data_fn()

    raw, labels = get_data(dtypes)

    ids, processed = data_process(raw, labels)
    split_and_save_data(ids, raw, processed)

    # gll = get_gll(dtypes[-1])
    # split_and_save_gll(gll)

    # psc = get_psc(PSCfns)
    # split_and_save_psc(psc)


if __name__ == '__main__':
    main()
