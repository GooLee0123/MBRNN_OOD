import logging
import os
import re

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv
import plotter
from loss import AnchorLoss
from model import Ensemble_E3 as E3
from optim import Optimizer


def _prepare_optim(Emodel):
    params = [p for p in Emodel.parameters() if p.requires_grad]
    # setting optimizer
    optimizer = Optimizer(
        torch.optim.Adam(
            params,
            lr=0.0008,
            betas=(0.5, 0.999),
            weight_decay=5e-5),
        max_grad_norm=5)
    # setting scheduler of optimizer for learning rate decay.
    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,
        patience=5,
        factor=0.5,
        min_lr=0.000001)
    optimizer.set_scheduler(scheduler)

    return optimizer


def _get_rdirs(opt):
    _univ_fn = 'univ_probs_in.npy'
    _spec_fn = 'spec_probs_in.npy'
    _ncls = opt.ncls
    _inquandtf = opt.quant_fd

    rdirs = []
    for var in opt.var:
        replaced = re.search('Gamma(.*?)_DCPW', _inquandtf).group(0)[:-5]
        _inquandtf = _inquandtf.replace(
            replaced, 'Gamma%s' % (str(var).replace('.', '_')))
        univ_rdir = os.path.join(_inquandtf, _univ_fn)
        spec_rdir = os.path.join(_inquandtf, _spec_fn)
        rdirs.append(univ_rdir)
        rdirs.append(spec_rdir)

    fbin = os.path.join('./bin_edges',
                        'galaxy_redshifts_%s-%s.txt'
                        % (_ncls, 'uniform'))

    return rdirs, fbin


def _load_probs(rdirs, opt, val=False, old_bins=None, new_bins=None):
    # Read probabilities for test
    probs = []
    for rdir in rdirs:
        if val:
            rdir = rdir.replace('_in.npy', '_vin.npy')

        if os.path.exists(rdir):
            results = np.load(rdir, allow_pickle=True).T
            log_msg = "Load probability outputs from %s\n" % rdir
            prob = results.astype(np.float32)
            probs.append(prob)
        else:
            log_msg = "%s doesn't exist." % rdir
            raise NotImplementedError(log_msg)

        logging.info(log_msg)

    probs = np.array(probs)  # [n_members, nbin, nsample]
    logging.info("probs shape is {}".format(str(probs.shape)))

    return probs


def equal_bin_ensemble(db, opt):
    test_set = db['eval_in'].dataset
    tX, tzspec = test_set.X.numpy(), test_set.z.numpy()

    # Read results
    rdirs_list = os.listdir(opt.quant_fd)
    rdirs_list = np.sort(rdirs_list)

    # Result file setting
    rdirs, fbin = _get_rdirs(opt)

    logging.info("Load bin from {}".format(fbin))
    bins = np.genfromtxt(fbin)
    binc = np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)])

    tprobs = _load_probs(rdirs, opt)
    vprobs = _load_probs(rdirs, opt, val=True)

    weights_dn = 'ensemble_weights'
    if not os.path.exists(weights_dn):
        os.makedirs(weights_dn)

    weights_fn = os.path.join(weights_dn, 'weights_NC'+str(opt.ncls))
    weights_fn += '.npy'

    if not os.path.exists(weights_fn):
        val_generator = db['val_in']
        val_set = val_generator.dataset

        probs_tensor = torch.from_numpy(vprobs).cuda()

        n_members = len(rdirs)

        Emodel = E3(n_members, ncls=opt.ncls)
        optim = _prepare_optim(Emodel)
        Emodel.train(True)
        Emodel.cuda()

        step = 0
        nbackp = 20000.
        if opt.batch_size < len(val_set):
            maxsteps = int(nbackp * (float(opt.batch_size)/len(val_set)))
        else:
            maxsteps = int(nbackp)
        logging.info("Train surrogate model for ensemble learning")
        for _ in range(maxsteps):
            optim.zero_grad()
            for Be, (_, local_zbin) in enumerate(val_generator):
                local_zbin = local_zbin.cuda()
                local_prob = probs_tensor[
                    :, :, Be*opt.batch_size:(Be+1)*opt.batch_size]

                ensemble_vprobs = Emodel(local_prob)

                loss = AnchorLoss(gamma=0)(ensemble_vprobs.transpose(0, 1), local_zbin)

                loss.backward()
                optim.step()

                if step % 100 == 0 and step != 0:
                    log_msg = "%sth ensemble step for " % step
                    log_msg += "optimal weights of models is done\n"
                    log_msg += "ensemble learning loss: %.5f" % loss.item()
                    logging.info(log_msg)
                step += 1

        Emodel.eval()

        # Point estimation for ensemble learning
        weights = Emodel.weights.cpu().detach().numpy()
        total_sum = np.tensordot(
            np.abs(weights), tprobs, axes=([0, 1], [0, 1]))  # [1, ndat]
        weights = np.abs(weights)/total_sum  # [nmem, ncls, ndat]

        np.save(weights_fn, weights)
        logging.info("Ensemble weights are saved at %s" % weights_fn)
    else:
        logging.info("Load Ensemble weights from %s" % weights_fn)
        weights = np.load(weights_fn)

    ensemble_tprobs = np.sum(weights*tprobs, axis=0)

    # save ensemble probs
    # if opt.test_option and opt.gamma == 0:
    #     bprob_fn = os.path.join(opt.eforvalfd, 'bin_prob.txt')
    #     np.savetxt(bprob_fn, ensemble_tprobs.T, fmt='%.6f')
    #     logging.info("[TEST]: Bin probabilities are saved at %s" % bprob_fn)

    # point-estimation, visualization and computing metrics
    ensemble_zphot = np.sum(ensemble_tprobs*binc.reshape(-1, 1), axis=0)
    ''' [ensemble_nbin, 1] '''

    outputs = np.hstack((ensemble_zphot.reshape(-1, 1), ensemble_tprobs.T))
    sfn = os.path.join(opt.ensemblefd, 'ensemble_results.txt')
    np.savetxt(sfn, outputs)
    logging.info("Ensemble results is saved at %s" % sfn)

    ensemble_zbin, _ = test_set.zbinning(tzspec, bins)

    reg_metric_inspection(tzspec, ensemble_zphot, opt)

    ezphot_fn = os.path.join(opt.ensemblefd, 'zphot.npy')
    np.save(ezphot_fn, ensemble_zphot)
    logging.info("Ensemble photo-z is saved at %s" % ezphot_fn)


def reg_metric_inspection(zspec, zphot, opt, dtype='ID'):
    reg_metricfn = 'reg_metric.csv'

    zspec = zspec.ravel()

    ztype = 'average'

    resid = (zspec-zphot)/(1+zspec)
    resid_abs = np.abs(resid)
    bias = np.abs(np.mean(resid))
    mar = np.mean(resid_abs)
    sig = np.std(resid)
    sig68 = np.percentile(resid_abs, 68)
    nmad = np.median(resid_abs)*1.4826
    rcat = np.sum(resid_abs > 0.15)/len(resid_abs)

    header = ["bias", "mar", "sig", "sig68", "nmad", "rcat"]
    outs = [bias, mar, sig, sig68, nmad, rcat]

    sfn = os.path.join(opt.ensemblefd,
                       ztype+'_'+dtype+'_'+reg_metricfn)
    with open(sfn, 'w') as outf:
        out = [outs[k] for k in range(len(outs))]
        writer = csv.writer(outf)
        writer.writerow(header)
        writer.writerow(out)
    logging.info("Regression metrics are saved at %s" % sfn)