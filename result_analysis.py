import csv
import logging
import os
import re

import numpy as np
import sklearn.metrics as metrics

import torch
import plotter
import model as Network

from optim import Optimizer
from loss import AnchorLoss

reg_metricfn = 'reg_metric.csv'
cls_metricfn = 'cls_metric.csv'
clb_metricfn = 'clb_metric.csv'
logger = logging.getLogger(__name__)


def reverse_dcp(dcp_label, dcp_ul, opt):
    dcp_in = dcp_label[0][:, 0]
    dcp_lo = dcp_label[1][:, 0]

    dcps = [dcp_in, dcp_lo, dcp_ul]
    dcp_np = np.hstack(dcps)

    if opt.psc:
        key = '_lpsc' if opt.psc_reg == 'low' else '_hpsc'
        quant_fd = opt.quant_fd.partition(key)[0]
        dcp_median_fn = os.path.join(quant_fd, 'dcp_median.txt')
        if not os.path.exists(dcp_median_fn):
            logging.warning("Run False-PSC first")
        dcp_median = np.genfromtxt(dcp_median_fn)
    else:
        dcp_median_fn = os.path.join(opt.quant_fd, 'dcp_median.txt')
        dcp_median = np.array([(dcp_np.min()+dcp_np.max())/2.])
        np.savetxt(dcp_median_fn, dcp_median)

    rdcps = []
    for dcp in dcps:
        dcp_translated = dcp - dcp_median
        dcp_reflected = -dcp_translated
        temp_rdcp = dcp_reflected + dcp_median
        rdcps.append(temp_rdcp)

    label_in = dcp_label[0][:, 1]
    label_lo = dcp_label[1][:, 1]

    rdcp_label = [np.vstack((rdcps[0], label_in)).T,
                  np.vstack((rdcps[1], label_lo)).T]
    rdcp_ul = rdcps[2]

    return rdcp_label, rdcp_ul


def reg_metric_inspection(zspec, zpreds, prefixs, opt,
                          dtype='ID'):
    zspec = zspec.ravel()

    ztypes = ['average', 'mode']
    for i, zphots in enumerate(zpreds):
        for j, zphot in enumerate(zphots):
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

            sfn = os.path.join(opt.quant_fd,
                               prefixs[j]+'_'+ztypes[i]
                               + '_'+dtype+'_'+reg_metricfn)
            with open(sfn, 'w') as outf:
                out = [outs[k] for k in range(len(outs))]
                writer = csv.writer(outf)
                writer.writerow(header)
                writer.writerow(out)
            logging.info("Regression metrics are saved at %s" % sfn)


def cls_metric_inspection(dcp_label, aucs, cals, opt):
    dcp_label = dcp_label[:-1]
    dcp_label = np.vstack(dcp_label)
    dcp = dcp_label.T[0]
    y_true = dcp_label.T[1]

    cls_threshold = (dcp.min()+dcp.max())/2.
    y_pred = np.where(dcp >= cls_threshold, 1, 0)

    # y_true = np.where(y_true == 2, 1, y_true)
    plotter.ConfusionMatrix(y_true, y_pred, opt)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f15_score = metrics.fbeta_score(y_true, y_pred, 1.5)

    header = ["accuracy", "precision", "recall",
              "f1.5_score", "roc_auc", "prc_auc"]
    outs = [accuracy, precision, recall, f15_score]+aucs

    sfn = os.path.join(opt.quant_fd, cls_metricfn)
    with open(sfn, 'w') as outf:
        out = [outs[k] for k in range(len(outs))]
        writer = csv.writer(outf)
        writer.writerow(header)
        writer.writerow(out)
    logging.info("classification metrics are saved at %s" % sfn)

    header = ["univ_ECE", "univ_MCE", "spec_ECE",
              "spec_MCE", "avg_ECE", "avg_MCE"]
    cfn = os.path.join(opt.quant_fd, clb_metricfn)
    with open(cfn, 'w') as outf:
        out = [cals[k] for k in range(len(cals))]
        writer = csv.writer(outf)
        writer.writerow(header)
        writer.writerow(out)
    logging.info("calibration metrics are saved at %s" % cfn)


def ea_reg_metric(zspec, zpreds, opt, dtype='ID', key='ensemble'):
    zspec = zspec.ravel()

    ztypes = ['average', 'mode']
    for i, zphot in enumerate(zpreds):
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

        sfn = os.path.join(opt.quant_fd,
                           key+'_'+opt.ecase+'_'+ztypes[i]
                           + '_'+dtype+'_'+reg_metricfn)
        with open(sfn, 'w') as outf:
            out = [outs[k] for k in range(len(outs))]
            writer = csv.writer(outf)
            writer.writerow(header)
            writer.writerow(out)
        logging.info(
            "Ensemble/avergae regression metrics are saved at %s" % sfn)


def read_losses(opt):
    loss_fns = os.listdir(opt.loss_fd)
    # cls_loss: [univ, spec]
    # dcp_loss: [id, ul]
    keys = ['train_cls', 'train_dcp', 'val_dcp']

    lfns = []
    for i, key in enumerate(keys):
        temp_fns = [os.path.join(opt.loss_fd, f) for f in loss_fns if key in f]
        temp_fns.sort(key=lambda f: int(re.sub('\D', '', f)))
        lfns.append(temp_fns)

    losses = [[], [], []]
    for i, lfn in enumerate(lfns):
        for fn in lfn:
            temp_loss = np.genfromtxt(fn).T
            if temp_loss.ndim == 1:
                temp_loss = temp_loss[:, np.newaxis]
            losses[i].append(temp_loss)
        losses[i] = np.hstack(losses[i])

    return losses


def model_ensemble(db, vpin, tpin, tplo, tpul, opt, train=False):
    def _accuracy(w_probs, zcls):
        pred_idx = w_probs.argmax(1).view(zcls.shape)
        correct = torch.sum(pred_idx == zcls).item()
        nsample = float(len(pred_idx))
        acc = correct/nsample
        return acc, correct, nsample

    def _compute_wprobs(weights, probs):
        if opt.ecase == 'case1':
            normed_weights = np.abs(weights)/np.sum(
                    np.abs(weights), axis=0)
            weighted_probs = np.sum(normed_weights*probs, axis=0)
        else:
            total_sum = np.tensordot(
                np.abs(weights), probs, axes=([0, 2], [0, 2]))
            normed_weights = np.abs(weights.transpose(0, 2, 1))/total_sum
            normed_weights = normed_weights.transpose(0, 2, 1)
            weighted_probs = np.sum(normed_weights*probs, axis=0)
        return weighted_probs

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
        return optimizer

    val_generator = db['val_in']
    val_set = val_generator.dataset
    binc = val_set.binc.numpy()

    vpin = torch.tensor(vpin).cuda()
    tpin = np.array(tpin)

    if opt.psc:
        key = '_lpsc' if opt.psc_reg == 'low' else '_hpsc'
        quant_fd = opt.quant_fd.partition(key)[0]
    else:
        quant_fd = opt.quant_fd

    sn = os.path.join(quant_fd, 'ensemble_weights_%s.npy') % opt.ecase
    if not os.path.exists(sn):
        Emodel = Network.Surrogate_Network(2, ncls=opt.ncls, case=opt.ecase)
        optim = _prepare_optim(Emodel)
        Emodel.train(True)
        Emodel.cuda()

        nbackp = 10000.
        step, tloss, tcrr, tnsample = 0, 0, 0, 0
        macc, mloss = 0, np.inf
        if opt.batch_size < len(val_set):
            maxsteps = int(nbackp * (float(opt.batch_size)/len(val_set)))
        else:
            maxsteps = int(nbackp)
        logging.info("Train surrogate model for ensemble learning (%s)"
                     % opt.ecase)
        for _ in range(maxsteps):
            optim.zero_grad()
            for Be, (_, local_zbin) in enumerate(val_generator):
                local_zbin = local_zbin.cuda()
                local_prob = vpin[
                    :, Be*opt.batch_size:(Be+1)*opt.batch_size, :]

                ensemble_vprobs = Emodel(local_prob)

                loss = AnchorLoss(gamma=0)(ensemble_vprobs, local_zbin)

                loss.backward()
                optim.step()

                acc, crr, nsample = _accuracy(ensemble_vprobs, local_zbin)

                tloss += loss.item()
                tcrr += crr
                tnsample += nsample

                step += 1
                if step % (nbackp/10) == 0:
                    log_msg = "%i%% ensemble step for " % (step//(nbackp/100))
                    log_msg += "optimal weights of models is done\n"
                    logging.info(log_msg)

            tloss /= (Be+1)
            tacc = tcrr/float(tnsample)
            if mloss >= tloss or tacc >= macc:
                log_msg = "Update weights\n"
                log_msg += "Validation average loss: %.3f, " % tloss
                log_msg += "Validation accuracy: %.3f" % tacc
                logging.info(log_msg)

                Emodel.eval()
                weights = Emodel.weights.cpu().detach().numpy()
                Emodel.train()

                if mloss >= tloss:
                    mloss = tloss
                if tacc >= macc:
                    macc = tacc
            tloss, tcrr, tnsample = 0, 0, 0

        np.save(sn, weights)
        logging.info("Ensemble weight is saved at %s" % sn)
    else:
        logging.info("Load ensemble weight from %s" % sn)
        weights = np.load(sn)

    weighted_probs_in = _compute_wprobs(weights, tpin)
    ez_avg_in = np.sum(weighted_probs_in*binc, axis=1)
    ez_mode_in = binc[weighted_probs_in.argmax(1)]

    weighted_probs_lo = _compute_wprobs(weights, tplo)
    ez_avg_lo = np.sum(weighted_probs_lo*binc, axis=1)
    ez_mode_lo = binc[weighted_probs_lo.argmax(1)]

    weighted_probs_ul = _compute_wprobs(weights, tpul)
    ez_avg_ul = np.sum(weighted_probs_ul*binc, axis=1)
    ez_mode_ul = binc[weighted_probs_ul.argmax(1)]

    return [ez_avg_in, ez_mode_in],\
           [ez_avg_lo, ez_mode_lo],\
           [ez_avg_ul, ez_mode_ul],\
           [weighted_probs_in, weighted_probs_lo, weighted_probs_ul]


def model_average(db, tpin, tplo, tpul, opt, train=False):
    def _accuracy(w_probs, zcls):
        pred_idx = w_probs.argmax(1).view(zcls.shape)
        correct = torch.sum(pred_idx == zcls).item()
        nsample = float(len(pred_idx))
        acc = correct/nsample
        return acc, correct, nsample

    def _compute_avg_probs(probs):
        summed = np.mean(probs, axis=0)
        return summed

    def _compute_photz(probs, binc):
        az_avg = np.sum(probs*binc, axis=1)
        az_mode = binc[probs.argmax(1)]
        return az_avg, az_mode

    val_generator = db['val_in']
    val_set = val_generator.dataset
    binc = val_set.binc.numpy()

    weighted_probs_in = _compute_avg_probs(tpin)
    az_avg_in, az_mode_in = _compute_photz(weighted_probs_in, binc)

    weighted_probs_lo = _compute_avg_probs(tplo)
    az_avg_lo, az_mode_lo = _compute_photz(weighted_probs_lo, binc)

    weighted_probs_ul = _compute_avg_probs(tpul)
    az_avg_ul, az_mode_ul = _compute_photz(weighted_probs_ul, binc)

    return [az_avg_in, az_mode_in],\
           [az_avg_lo, az_mode_lo],\
           [az_avg_ul, az_mode_ul],\
           [weighted_probs_in, weighted_probs_lo, weighted_probs_ul]


def unsup_visual_inspection(db, zspec_in, zspec_lo,
                            ez_in, ez_lo, ez_ul,
                            az_in, az_lo, az_ul,
                            z_in, z_lo, z_ul,
                            dcp_label, dcp_ul,
                            univ_probs, spec_probs,
                            eprobs, aprobs,
                            prefixs, opt):
    # if not opt.psc:
    #     losses = read_losses(opt)
    #     plotter.LossVariation(*losses, opt)

    dcp_label = dcp_label[:-1]

    zcls = db['eval_in'].dataset.zcls.numpy()
    ucal = plotter.CalibrationPlot(univ_probs[0], zcls, opt)
    scal = plotter.CalibrationPlot(spec_probs[0], zcls,
                                   opt, key='specific')
    acal = plotter.CalibrationPlot(aprobs[0], zcls,
                                   opt, key='average')
    cals = ucal+scal+acal

    # plotter.RedshiftDistribution(zspec_in, z_in, opt, key="ID")
    # plotter.RedshiftDistribution(zspec_lo, z_lo, opt, key="LO")

    # binc = db['eval_in'].dataset.binc.numpy()
    # plotter.ProbDensityDistribution(univ_probs, binc, dcp_label,
    #                                 dcp_ul, opt, model='universal')
    # plotter.ProbDensityDistribution(spec_probs, binc, dcp_label,
    #                                 dcp_ul, opt, model='specific')
    # plotter.AE_ProbDensityDistribution(eprobs, binc,
    #                                    dcp_label, dcp_ul, opt)
    # plotter.AE_ProbDensityDistribution(aprobs, binc,
    #                                    dcp_label, dcp_ul, opt,
    #                                    key='average')

    plotter.DiscriminatorNumberDensity(dcp_label, dcp_ul, opt, "Discrepancy")

    roc_auc, prc_auc = plotter.ROC_PR_Curves(dcp_label, opt)

    # plotter.PCAScattergram(db, dcp_label, dcp_ul, opt)
    # plotter.PCAScattergram(db, dcp_label, dcp_ul, opt, scolor='dcp')

    # plotter.Mollweide_GLL(dcp_ul, opt)
    # plotter.GLatPDF(dcp_ul, opt)
    if '0_10' in opt.ul_prefix or '100_110' in opt.ul_prefix or '170_180' in opt.ul_prefix:
        # plotter.PSC_DCP_scatter(dcp_ul, opt)
        plotter.PSCPDF(dcp_ul, opt)
    # plotter.LPSC_scattergram(db, dcp_ul, opt)
    # plotter.LPSC_histogram(db, dcp_ul, opt)

    # if 'combined_usample' in opt.ul_prefix:
    #     plotter.ColorColorDensity(db, dcp_label, dcp_ul, opt)
    # plotter.ColorColorDensityDiff(db, dcp_label, dcp_ul, opt)
    # plotter.ColorColorContour(db, dcp_label, dcp_ul, opt)
    # plotter.ColorColorScattergram(db, dcp_label, dcp_ul, opt)

    # plotter.PhotRedshiftColorScattergram(db, dcp_label, dcp_ul,
    #                                      z_in, z_lo, z_ul, opt)
    # plotter.PhotRedshiftColorScattergram(db, dcp_label, dcp_ul,
    #                                      z_in, z_lo, z_ul, opt,
    #                                      model='specific')

    # if opt.ind != 'star' and not opt.psc:
    #     plotter.Scattergram(zspec_in, z_in, dcp_label, dcp_ul, prefixs, opt)
    #     plotter.Scattergram(zspec_in, z_in, dcp_label, dcp_ul, prefixs, opt,
    #                         col='dcp')
        # plotter.AE_Scattergram(zspec_in, ez_in, dcp_label, dcp_ul, opt)
        # plotter.AE_Scattergram(zspec_in, ez_in, dcp_label, dcp_ul, opt,
        #                        col='dcp')
        # plotter.AE_Scattergram(zspec_in, az_in, dcp_label, dcp_ul, opt,
        #                        key='average')
        # plotter.AE_Scattergram(zspec_in, az_in, dcp_label, dcp_ul, opt,
        #                        col='dcp', key='average')

        # plotter.Scattergram(zspec_lo, z_lo, dcp_label, dcp_ul, prefixs, opt,
        #                     dtype='LO')
        # plotter.Scattergram(zspec_lo, z_lo, dcp_label, dcp_ul, prefixs, opt,
        #                     dtype='LO', col='dcp')
        # plotter.AE_Scattergram(zspec_lo, ez_lo, dcp_label, dcp_ul, opt,
        #                        dtype='LO')
        # plotter.AE_Scattergram(zspec_lo, ez_lo, dcp_label, dcp_ul, opt,
        #                        dtype='LO', col='dcp')
        # plotter.AE_Scattergram(zspec_lo, az_lo, dcp_label, dcp_ul, opt,
        #                        dtype='LO', key='average')
        # plotter.AE_Scattergram(zspec_lo, az_lo, dcp_label, dcp_ul, opt,
        #                        dtype='LO', col='dcp', key='average')

        # plotter.SpecRedshiftColorScattergram(db, dcp_label, dcp_ul, opt)

    return [roc_auc, prc_auc], cals


def unsup_analysis(db, zspec_in, zspec_lo, z_in, z_lo, z_ul,
                   dcp_label, dcp_ul, univ_probs, spec_probs,
                   vprobs_in, opt):
    tprobs_in = [univ_probs[0], spec_probs[0]]
    tprobs_lo = [univ_probs[1], spec_probs[1]]
    tprobs_ul = [univ_probs[2], spec_probs[2]]
    ez_in, ez_lo, ez_ul, eprobs = model_ensemble(db,
                                                 vprobs_in,
                                                 tprobs_in,
                                                 tprobs_lo,
                                                 tprobs_ul,
                                                 opt, train=True)
    ea_reg_metric(zspec_in, ez_in, opt)

    az_in, az_lo, az_ul, aprobs = model_average(db,
                                                tprobs_in,
                                                tprobs_lo,
                                                tprobs_ul,
                                                opt)

    ea_reg_metric(zspec_in, az_in, opt, key='average')

    prefixs = ['universal', 'specific']
    reg_metric_inspection(zspec_in, z_in, prefixs, opt)
    reg_metric_inspection(zspec_lo, z_lo, prefixs, opt, dtype='LO')
    # dcp_label, dcp_ul = reverse_dcp(dcp_label, dcp_ul, opt)
    aucs, cals = unsup_visual_inspection(db, zspec_in, zspec_lo,
                                         ez_in, ez_lo, ez_ul,
                                         az_in, az_lo, az_ul,
                                         z_in, z_lo, z_ul,
                                         dcp_label, dcp_ul,
                                         univ_probs, spec_probs,
                                         eprobs, aprobs,
                                         prefixs, opt)

    cls_metric_inspection(dcp_label, aucs, cals, opt)


def compute_rlkhd(ll_labels):
    rll_labels = []
    for ll_label in ll_labels:
        smt_ll = ll_label.T[0]
        bck_ll = ll_label.T[1]

        rll = -smt_ll + bck_ll

        rll_label = np.vstack((rll, ll_label.T[-1]))
        rll_labels.append(rll_label.T)

    return rll_labels


def lkhd_visual_inspection(zspec, zphots, rll_labels, ll_labels, prefixs, opt):
    plotter.DiscriminatorNumberDensity(ll_labels, opt, "Likelihood")
    plotter.DiscriminatorNumberDensity(rll_labels, opt, "Likelihood Ratio")
    plotter.lkhd_ROCCurves(rll_labels, ll_labels, opt)

    if opt.ind != 'star':
        plotter.Scattergram(zspec, zphots, prefixs, opt)


def lkhd_analysis(zspec, zphots, ll_labels, opt):
    prefixs = ['semantic', 'background']
    rll_labels = compute_rlkhd(ll_labels)
    reg_metric_inspection(zspec, zphots, prefixs, opt)
    lkhd_visual_inspection(zspec, zphots, rll_labels, ll_labels, prefixs, opt)
