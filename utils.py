import shutil
import logging
import os
import time
import copy

import numpy as np
import torch

from checkpoint import Checkpoint
from optim import Optimizer


def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.affine = False
        m.track_running_stack = False
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


def set_bn_train(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.train()
        m.affine = True
        m.track_running_stack = True
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)


def loss_containers(opt):
    ldic = {}

    # t: train, v: val, p: print

    if opt.method == 'unsup':
        ldic['tuniv'] = []
        ldic['ptuniv'] = [0., 0.]
        ldic['pvuniv'] = 0.

        ldic['tspec'] = []
        ldic['ptspec'] = [0., 0.]
        ldic['pvspec'] = 0.

        ldic['tdcp_in'] = []
        ldic['vdcp_in'] = []
        ldic['ptdcp_in'] = [0., 0.]
        ldic['pvdcp_in'] = 0.

        ldic['tdcp_ul'] = []
        ldic['vdcp_ul'] = []
        ldic['ptdcp_ul'] = [0., 0.]
        ldic['pvdcp_ul'] = 0.
    elif opt.method == 'lkhd':
        ldic['tsmt'] = []
        ldic['ptsmt'] = [0., 0.]
        ldic['pvsmt'] = 0.

        ldic['tbck'] = []
        ldic['ptbck'] = [0., 0.]
        ldic['pvbck'] = 0.

    return ldic


def unsup_train_epoch(models, optim, db, step, epoch, do_finetune,
                      post_tune, finetune_tol, best_vcls, best_vdcp, opt):
    finetune_in = 0
    ldic = loss_containers(opt)

    # train_switch = False
    models = models.cuda()
    models = models.train()
    for Be, (local_batch_in, local_zbin_in) in enumerate(db['train_in']):
        local_batch_in = local_batch_in.cuda()
        local_zbin_in = local_zbin_in.cuda()

        optim.zero_grad()

        if do_finetune:
            if finetune_in < 2:
                local_batch_ul, _ = next(db['train_ul_cycle'])
                local_batch_ul = local_batch_ul.cuda()

                local_batch = torch.cat((local_batch_in, local_batch_ul), 0)

                nin = local_batch_in.size(0)
            else:
                local_batch = local_batch_in
        else:
            local_batch = local_batch_in

        univ_probs, spec_probs = models(local_batch)

        if do_finetune:
            if finetune_in < 2:
                univ_probs_in = univ_probs[:nin]
                univ_probs_ul = univ_probs[nin:]

                spec_probs_in = spec_probs[:nin]
                spec_probs_ul = spec_probs[nin:]
            else:
                univ_probs_in = univ_probs
                spec_probs_in = spec_probs
        else:
            univ_probs_in = univ_probs
            spec_probs_in = spec_probs

        univ_cls = opt.cls_criterion(
            univ_probs_in, local_zbin_in)
        spec_cls = opt.cls_criterion(
            spec_probs_in, local_zbin_in)

        ldic['ptuniv'][0] += univ_cls.item()
        ldic['ptspec'][0] += spec_cls.item()
        ldic['ptuniv'][1] += 1
        ldic['ptspec'][1] += 1

        tcls_loss = univ_cls + spec_cls

        if do_finetune:
            if finetune_in < 2:
                tdcp_in = opt.dcp_criterion(univ_probs_in, spec_probs_in)
                tdcp_ul = opt.dcp_criterion(univ_probs_ul, spec_probs_ul)
                tcls_loss += opt.dcp_weight*tdcp_ul

                ldic['ptdcp_in'][0] += tdcp_in.item()
                ldic['ptdcp_ul'][0] += tdcp_ul.item()
                ldic['ptdcp_in'][1] += 1
                ldic['ptdcp_ul'][1] += 1

                finetune_in += 1
            else:
                finetune_in = 0

        tcls_loss.backward()
        optim.step()

        # print training information
        if step != 0 and step % opt.pevery == 0:
            # for log messages
            avg_univ_cls = \
                ldic['ptuniv'][0]/ldic['ptuniv'][1]
            avg_spec_cls = \
                ldic['ptspec'][0]/ldic['ptspec'][1]

            ldic['ptuniv'] = [0., 0.]
            ldic['ptspec'] = [0., 0.]

            ldic['tuniv'].append(avg_univ_cls)
            ldic['tspec'].append(avg_spec_cls)

            if do_finetune:
                avg_dcp_in = ldic['ptdcp_in'][0]/ldic['ptdcp_in'][1]
                avg_dcp_ul = ldic['ptdcp_ul'][0]/ldic['ptdcp_ul'][1]

                ldic['ptdcp_in'] = [0., 0.]
                ldic['ptdcp_ul'] = [0., 0.]

                ldic['tdcp_in'].append(avg_dcp_in)
                ldic['tdcp_ul'].append(avg_dcp_ul)

                for g in optim.param_groups():
                    g['lr'] = opt.finetune_lr

            for param in optim.param_groups():
                lr = param['lr']  # learning rate.

            # log messages
            log_msg = "(fine-tune) " if do_finetune else "(post-tune) "\
                if post_tune else ""
            log_msg += "cls loss (univ): %.3f, " % avg_univ_cls
            log_msg += "cls loss (spec): %.3f" % avg_spec_cls
            if do_finetune:
                log_msg += ", dcp loss (ID): %.3f" % avg_dcp_in
                log_msg += ", dcp loss (UL): %.3f" % avg_dcp_ul

            log_msg += ", learning rate: %.5f" % lr
            logging.info(log_msg)

        # validation
        if step != 0 and step % opt.vevery == 0:
            models = models.eval()
            with torch.no_grad():
                vstep_in, vcls = 0, 0
                vuniv_correct, vspec_correct = 0, 0
                for vlocal_batch_in, vlocal_zbin_in in db['val_in']:
                    # classification
                    vlocal_batch_in = vlocal_batch_in.cuda()
                    vlocal_zbin_in = vlocal_zbin_in.cuda()

                    vuniv_probs_in, vspec_probs_in = models(vlocal_batch_in)

                    univ_cls = opt.cls_criterion(vuniv_probs_in,
                                                 vlocal_zbin_in)
                    spec_cls = opt.cls_criterion(vspec_probs_in,
                                                 vlocal_zbin_in)

                    vcls += univ_cls + spec_cls

                    # discrepancy
                    vdcp_in = opt.dcp_criterion(vuniv_probs_in,
                                                vspec_probs_in)

                    ldic['pvuniv'] += univ_cls
                    ldic['pvspec'] += spec_cls

                    ldic['pvdcp_in'] += vdcp_in

                    # get the index of the maximum probability
                    vuniv_pred = vuniv_probs_in.data.max(1, keepdim=True)[1]
                    vspec_pred = vspec_probs_in.data.max(1, keepdim=True)[1]
                    vuniv_correct += vuniv_pred.eq(
                        vlocal_zbin_in.data.view_as(vuniv_pred)).sum()
                    vspec_correct += vspec_pred.eq(
                        vlocal_zbin_in.data.view_as(vspec_pred)).sum()

                    vstep_in += 1

                nsample_ul = 100000
                temp_db_ul = copy.deepcopy(db['val_ul'])
                randidx = torch.randint(
                    0, len(temp_db_ul.dataset), (nsample_ul,))
                temp_db_ul.dataset.X = temp_db_ul.dataset.X[randidx]
                temp_db_ul.dataset.y = temp_db_ul.dataset.y[randidx]
                temp_db_ul.dataset.update_len(nsample_ul)
                vstep_ul = 0
                for vlocal_batch_ul, _ in temp_db_ul:
                    vlocal_batch_ul = vlocal_batch_ul.cuda()

                    vuniv_probs_ul, vspec_probs_ul = models(vlocal_batch_ul)

                    vdcp_loss_ul = opt.dcp_criterion(vuniv_probs_ul,
                                                     vspec_probs_ul)

                    ldic['pvdcp_ul'] += vdcp_loss_ul
                    vstep_ul += 1

                avg_vdcp_in = ldic['pvdcp_in']/vstep_in
                avg_vdcp_ul = ldic['pvdcp_ul']/vstep_ul

                ldic['pvuniv'] = 0.
                ldic['pvspec'] = 0.
                ldic['pvdcp_in'] = 0.
                ldic['pvdcp_ul'] = 0.

                ldic['vdcp_in'].append(avg_vdcp_in)
                ldic['vdcp_ul'].append(avg_vdcp_ul)

                avg_vloss = vcls/vstep_in
                cls_ckpt_cond = avg_vloss < best_vcls

                vdcp_diff = \
                    avg_vdcp_in-avg_vdcp_ul
                dcp_ckpt_cond = \
                    do_finetune and vdcp_diff > best_vdcp
                if cls_ckpt_cond or dcp_ckpt_cond:
                    if cls_ckpt_cond:
                        best_vcls = avg_vloss
                        keywords = ("cls loss", "smaller", "minimum")
                    if dcp_ckpt_cond:
                        best_vdcp = vdcp_diff
                        keywords = ("dcp diff", "bigger", "maximum")

                    msg = "Validation %s being %s than previous %s" % \
                        keywords
                    msg += ", checkpoint save condition is met."

                    logging.info(msg)

                    checkpoint = Checkpoint(
                        step, epoch, models, optim, opt=opt)
                    checkpoint.save(do_finetune, post_tune)
                if cls_ckpt_cond or dcp_ckpt_cond:
                    finetune_tol = 0
                else:
                    finetune_tol += 1

                if opt.finetune and \
                        not post_tune and \
                        not do_finetune and \
                        finetune_tol > opt.finetune_patience:
                    logging.info("Start to train with unlabeled data")
                    do_finetune = True
                    finetune_tol = 0

                    if opt.optim_change:
                        optimizer = torch.optim.SGD(
                            list(models[0].parameters()) +
                            list(models[1].parameters()),
                            lr=0.1,
                            weight_decay=5e-5)

                        optim = Optimizer(
                            optimizer,
                            max_grad_norm=5.)
                if do_finetune and finetune_tol > opt.posttune_patience:
                    logging.info("Stop fine-tuning and start post-tuning")
                    do_finetune = False
                    post_tune = True

                # for validation log message
                vdat_len = len(db['val_in'].dataset)
                universal_acc = float(vuniv_correct)/vdat_len
                specific_acc = float(vspec_correct)/vdat_len

                log_msg = \
                    "Validation universal accuracy: %i/%i (%.6f)\n" % \
                    (vuniv_correct, vdat_len, universal_acc)
                log_msg += \
                    "Validation specific accuracy: %i/%i (%.6f)\n" % \
                    (vspec_correct, vdat_len, specific_acc)
                log_msg += "current validation cls loss: %.5f, " % \
                    avg_vloss
                log_msg += "minimum cls loss: %.5f\n" % best_vcls
                log_msg += "current validation ID-dcp loss: %.5f, " % \
                    avg_vdcp_in
                log_msg += "OOD-dcp loss: %.5f\n" % \
                    avg_vdcp_ul
                logging.info(log_msg)

            models = models.train()
        step += 1

    if epoch >= opt.lr_decay_epoch:
        optim.update(avg_vloss, epoch)

    return best_vcls, best_vdcp, do_finetune, post_tune,\
        finetune_tol, ldic, step, optim, db


def unsup_train(db, models, optim, opt, counts=[0, 0]):
    logging.info(">> Unsupervised train")

    sepoch, step = counts
    stime = time.time()  # start time
    bcv = np.inf  # minimum validation loss placeholder
    bdv = 0  # maximum discrepancy difference between in and ul data
    do_finetune = False
    post_tune = True if opt.resume else False
    finetune_tol = 0
    save_losses(opt, delete=True)
    for epoch in range(sepoch, opt.max_training_epoch):
        args = unsup_train_epoch(models, optim, db, step, epoch, do_finetune,
                                 post_tune, finetune_tol, bcv, bdv, opt)
        bcv, bdv, do_finetune, post_tune, finetune_tol, ldic, step, optim, db = args

        log_msg = "Finished epoch %s/%s" % (epoch, opt.max_training_epoch)
        logging.info(log_msg)

        save_losses(opt, ldic=ldic, epoch=epoch)

    etime = time.time()  # end time
    dur = etime - stime  # training time
    logging.info("Training is done. Took %.3fh" % (dur/3600.))


def unsup_evaluate(db, models, opt):
    logging.info(">> Unsupervised evaluate")

    # model setting
    models, _, _ = set_loaded_model(models, opt)

    models = models.eval().zero_grad().cuda()

    val_set_in = db['val_in'].dataset
    eval_set_in = db['eval_in'].dataset
    eval_set_lo = db['eval_lo'].dataset
    eval_set_ul = db['eval_ul'].dataset
    zspec_in = eval_set_in.z.numpy()
    zspec_lo = eval_set_lo.z.numpy()

    vinlen = len(val_set_in)
    inlen = len(eval_set_in)
    lolen = len(eval_set_lo)
    ullen = len(eval_set_ul)

    univ_ncorrect = 0  # redshift classification
    spec_ncorrect = 0
    dcp_id_in = []
    univ_probs_vin_stacked = torch.empty(vinlen, opt.ncls).cuda()
    spec_probs_vin_stacked = torch.empty(vinlen, opt.ncls).cuda()

    univ_probs_in_stacked = torch.empty(inlen, opt.ncls).cuda()
    spec_probs_in_stacked = torch.empty(inlen, opt.ncls).cuda()
    univ_zphot_in_stacked = torch.empty(inlen).cuda()
    spec_zphot_in_stacked = torch.empty(inlen).cuda()
    univ_zmode_in_stacked = torch.empty(inlen).cuda()
    spec_zmode_in_stacked = torch.empty(inlen).cuda()

    univ_probs_lo_stacked = torch.empty(lolen, opt.ncls).cuda()
    spec_probs_lo_stacked = torch.empty(lolen, opt.ncls).cuda()
    univ_zphot_lo_stacked = torch.empty(lolen).cuda()
    spec_zphot_lo_stacked = torch.empty(lolen).cuda()
    univ_zmode_lo_stacked = torch.empty(lolen).cuda()
    spec_zmode_lo_stacked = torch.empty(lolen).cuda()

    univ_probs_ul_stacked = torch.empty(ullen, opt.ncls).cuda()
    spec_probs_ul_stacked = torch.empty(ullen, opt.ncls).cuda()
    univ_zphot_ul_stacked = torch.empty(ullen).cuda()
    spec_zphot_ul_stacked = torch.empty(ullen).cuda()
    univ_zmode_ul_stacked = torch.empty(ullen).cuda()
    spec_zmode_ul_stacked = torch.empty(ullen).cuda()

    with torch.no_grad():
        for bepoch, (local_batch_in, local_zbin_in) in \
                enumerate(db['val_in']):
            binc = eval_set_in.binc.cuda()
            local_batch_in = local_batch_in.cuda()
            local_zbin_in = local_zbin_in.cuda()

            # input into model
            univ_probs_in, spec_probs_in = models(local_batch_in)

            store_sidx = bepoch*opt.batch_size
            store_eidx = store_sidx+opt.batch_size

            univ_probs_vin_stacked[store_sidx:store_eidx] = univ_probs_in
            spec_probs_vin_stacked[store_sidx:store_eidx] = spec_probs_in

        univ_probs_vin = univ_probs_vin_stacked.cpu().detach().numpy()
        spec_probs_vin = spec_probs_vin_stacked.cpu().detach().numpy()

        vprobs_in = [univ_probs_vin, spec_probs_vin]

        for bepoch, (local_batch_in, local_zbin_in) in \
                enumerate(db['eval_in']):
            binc = eval_set_in.binc.cuda()
            local_batch_in = local_batch_in.cuda()
            local_zbin_in = local_zbin_in.cuda()

            # input into model
            univ_probs_in, spec_probs_in = models(local_batch_in)

            dcp_in = opt.dcp_criterion(univ_probs_in, spec_probs_in)

            dcp_id_in.append(dcp_in.cpu().detach().numpy().ravel())

            # get the index of the maximum log-probability
            univ_pred_in = \
                univ_probs_in.data.max(1, keepdim=True)[1]
            spec_pred_in = \
                spec_probs_in.data.max(1, keepdim=True)[1]
            univ_correct = univ_pred_in.eq(
                local_zbin_in.data.view_as(univ_pred_in))
            spec_correct = spec_pred_in.eq(
                local_zbin_in.data.view_as(spec_pred_in))
            univ_ncorrect += univ_correct.cpu().sum()
            spec_ncorrect += spec_correct.cpu().sum()

            univ_zphot_in = torch.sum(
                univ_probs_in*binc, dim=1).view(-1)
            spec_zphot_in = torch.sum(
                spec_probs_in*binc, dim=1).view(-1)

            univ_zmode_in = binc[univ_pred_in].view(-1)
            spec_zmode_in = binc[spec_pred_in].view(-1)

            store_sidx = bepoch*opt.batch_size
            store_eidx = store_sidx+opt.batch_size

            univ_probs_in_stacked[store_sidx:store_eidx] = univ_probs_in
            spec_probs_in_stacked[store_sidx:store_eidx] = spec_probs_in
            univ_zphot_in_stacked[store_sidx:store_eidx] = univ_zphot_in
            spec_zphot_in_stacked[store_sidx:store_eidx] = spec_zphot_in
            univ_zmode_in_stacked[store_sidx:store_eidx] = univ_zmode_in
            spec_zmode_in_stacked[store_sidx:store_eidx] = spec_zmode_in

        univ_probs_in_stacked = univ_probs_in_stacked.cpu().detach()
        spec_probs_in_stacked = spec_probs_in_stacked.cpu().detach()
        univ_zphot_in_stacked = univ_zphot_in_stacked.cpu().detach()
        spec_zphot_in_stacked = spec_zphot_in_stacked.cpu().detach()
        univ_zmode_in_stacked = univ_zmode_in_stacked.cpu().detach()
        spec_zmode_in_stacked = spec_zmode_in_stacked.cpu().detach()

        dcp_id_in = np.hstack(dcp_id_in)
        dcp_label_in = np.vstack((dcp_id_in, np.zeros(len(dcp_id_in))))

        deval_len = len(eval_set_in)
        univ_correct = float(univ_ncorrect)/deval_len
        spec_correct = float(spec_ncorrect)/deval_len
        log_msg = "universal eval set accuracy: %i/%i (%.6f), " % \
            (univ_ncorrect, deval_len, univ_correct)
        log_msg += "specific eval set accuracy: %i/%i (%.6f)\n" % \
            (spec_ncorrect, deval_len, spec_correct)
        logging.info(log_msg)

        univ_probs_in = univ_probs_in_stacked.numpy()
        spec_probs_in = spec_probs_in_stacked.numpy()

        univ_zphots_in = univ_zphot_in_stacked.numpy()
        spec_zphots_in = spec_zphot_in_stacked.numpy()
        zphots_in = np.vstack((univ_zphots_in, spec_zphots_in))

        univ_zmodes_in = univ_zmode_in_stacked.numpy().ravel()
        spec_zmodes_in = spec_zmode_in_stacked.numpy().ravel()
        zmodes_in = np.vstack((univ_zmodes_in, spec_zmodes_in))

        dcp_ul = []
        for bepoch, (local_batch_ul, _) in \
                enumerate(db['eval_ul']):
            local_batch_ul = local_batch_ul.cuda()

            univ_probs_ul, spec_probs_ul = models(local_batch_ul)

            dcp_loss_ul = opt.dcp_criterion(univ_probs_ul, spec_probs_ul)

            dcp_ul.append(
                dcp_loss_ul.cpu().detach().numpy().ravel())

            univ_pred_ul = \
                univ_probs_ul.data.max(1, keepdim=True)[1]
            spec_pred_ul = \
                spec_probs_ul.data.max(1, keepdim=True)[1]

            univ_zphot_ul = torch.sum(
                univ_probs_ul*binc, dim=1).view(-1)
            spec_zphot_ul = torch.sum(
                spec_probs_ul*binc, dim=1).view(-1)

            univ_zmode_ul = binc[univ_pred_ul].view(-1)
            spec_zmode_ul = binc[spec_pred_ul].view(-1)

            store_sidx = bepoch*opt.batch_size
            store_eidx = store_sidx+opt.batch_size

            univ_probs_ul_stacked[store_sidx:store_eidx] = univ_probs_ul
            spec_probs_ul_stacked[store_sidx:store_eidx] = spec_probs_ul
            univ_zphot_ul_stacked[store_sidx:store_eidx] = univ_zphot_ul
            spec_zphot_ul_stacked[store_sidx:store_eidx] = spec_zphot_ul
            univ_zmode_ul_stacked[store_sidx:store_eidx] = univ_zmode_ul
            spec_zmode_ul_stacked[store_sidx:store_eidx] = spec_zmode_ul

        univ_probs_ul_stacked = univ_probs_ul_stacked.cpu().detach()
        spec_probs_ul_stacked = spec_probs_ul_stacked.cpu().detach()
        univ_zphot_ul_stacked = univ_zphot_ul_stacked.cpu().detach()
        spec_zphot_ul_stacked = spec_zphot_ul_stacked.cpu().detach()
        univ_zmode_ul_stacked = univ_zmode_ul_stacked.cpu().detach()
        spec_zmode_ul_stacked = spec_zmode_ul_stacked.cpu().detach()

        univ_probs_ul = univ_probs_ul_stacked.numpy()
        spec_probs_ul = spec_probs_ul_stacked.numpy()

        univ_zphots_ul = univ_zphot_ul_stacked.numpy()
        spec_zphots_ul = spec_zphot_ul_stacked.numpy()
        zphots_ul = np.vstack((univ_zphots_ul, spec_zphots_ul))

        univ_zmodes_ul = univ_zmode_ul_stacked.numpy().ravel()
        spec_zmodes_ul = spec_zmode_ul_stacked.numpy().ravel()
        zmodes_ul = np.vstack((univ_zmodes_ul, spec_zmodes_ul))

        dcp_ul = np.hstack(dcp_ul)
        dcp_label_ul = np.vstack((dcp_ul, np.ones(len(dcp_ul))+1))

        # dcp_id_lo = []
        dcp_ood_lo = []
        for bepoch, (local_batch_lo, _) in \
                enumerate(db['eval_lo']):
            local_batch_lo = local_batch_lo.cuda()

            univ_probs_lo, spec_probs_lo = models(local_batch_lo)

            dcp_loss_lo = opt.dcp_criterion(univ_probs_lo, spec_probs_lo)

            dcp_ood_lo.append(
                dcp_loss_lo.cpu().detach().numpy().ravel())

            univ_pred_lo = \
                univ_probs_lo.data.max(1, keepdim=True)[1]
            spec_pred_lo = \
                spec_probs_lo.data.max(1, keepdim=True)[1]

            univ_zphot_lo = torch.sum(
                univ_probs_lo*binc, dim=1).view(-1)
            spec_zphot_lo = torch.sum(
                spec_probs_lo*binc, dim=1).view(-1)

            univ_zmode_lo = binc[univ_pred_lo].view(-1)
            spec_zmode_lo = binc[spec_pred_lo].view(-1)

            store_sidx = bepoch*opt.batch_size
            store_eidx = store_sidx+opt.batch_size

            univ_probs_lo_stacked[store_sidx:store_eidx] = univ_probs_lo
            spec_probs_lo_stacked[store_sidx:store_eidx] = spec_probs_lo
            univ_zphot_lo_stacked[store_sidx:store_eidx] = univ_zphot_lo
            spec_zphot_lo_stacked[store_sidx:store_eidx] = spec_zphot_lo
            univ_zmode_lo_stacked[store_sidx:store_eidx] = univ_zmode_lo
            spec_zmode_lo_stacked[store_sidx:store_eidx] = spec_zmode_lo

        univ_probs_lo_stacked = univ_probs_lo_stacked.cpu().detach()
        spec_probs_lo_stacked = spec_probs_lo_stacked.cpu().detach()
        univ_zphot_lo_stacked = univ_zphot_lo_stacked.cpu().detach()
        spec_zphot_lo_stacked = spec_zphot_lo_stacked.cpu().detach()
        univ_zmode_lo_stacked = univ_zmode_lo_stacked.cpu().detach()
        spec_zmode_lo_stacked = spec_zmode_lo_stacked.cpu().detach()

        dcp_ood_lo = np.hstack(dcp_ood_lo)
        dcp_label_ood_lo = np.vstack((dcp_ood_lo, np.ones(len(dcp_ood_lo))))

        univ_probs_lo = univ_probs_lo_stacked.numpy()
        spec_probs_lo = spec_probs_lo_stacked.numpy()

        univ_zphots_lo = univ_zphot_lo_stacked.numpy()
        spec_zphots_lo = spec_zphot_lo_stacked.numpy()
        zphots_lo = np.vstack((univ_zphots_lo, spec_zphots_lo))

        univ_zmodes_lo = univ_zmode_lo_stacked.numpy().ravel()
        spec_zmodes_lo = spec_zmode_lo_stacked.numpy().ravel()
        zmodes_lo = np.vstack((univ_zmodes_lo, spec_zmodes_lo))

    # dcp_label_id_lo.T
    dcp_label = [dcp_label_in.T, dcp_label_ood_lo.T, dcp_label_ul.T]

    save_outputs = [zphots_in.T, zphots_lo.T, zphots_ul.T]+dcp_label
    save_results(save_outputs, opt)

    # save_results(np.array([univ_probs_in, spec_probs_in]), opt, 'probs_id')
    save_results(np.array([univ_probs_lo, spec_probs_lo]), opt, 'probs_lo')
    # save_results(np.array([univ_probs_ul, spec_probs_ul]), opt, 'probs_ul')
    # quit()

    z_in = [zphots_in, zmodes_in]
    z_lo = [zphots_lo, zmodes_lo]
    z_ul = [zphots_ul, zmodes_ul]
    univ_probs = [univ_probs_in, univ_probs_lo, univ_probs_ul]
    spec_probs = [spec_probs_in, spec_probs_lo, spec_probs_ul]

    # save_results(univ_probs_in, opt, 'univ_probs_in.npy')
    # save_results(spec_probs_in, opt, 'spec_probs_in.npy')

    # save_results(univ_probs_vin, opt, 'univ_probs_vin.npy')
    # save_results(spec_probs_vin, opt, 'spec_probs_vin.npy')

    return zspec_in, zspec_lo, z_in, z_lo, z_ul, \
        dcp_label, dcp_ul, univ_probs, spec_probs, vprobs_in


def lkhd_train_epoch(models, optim, db, step, epoch, bcv, opt):
    ldic = loss_containers(opt)

    models = models.cuda().train()
    for Be, (local_batch_in, local_zbin_in) \
            in enumerate(db['train_in']):
            # in tqdm(enumerate(db['train_in']), total=len(db['train_in'])):
        local_batch_in = local_batch_in.cuda()
        local_zbin_in = local_zbin_in.cuda()

        optim.zero_grad()

        smt_probs, bck_probs = models(local_batch_in)

        smt_cls = opt.cls_criterion(
            smt_probs, local_zbin_in)
        bck_cls = opt.cls_criterion(
            bck_probs, local_zbin_in)

        ldic['ptsmt'][0] += smt_cls.item()
        ldic['ptbck'][0] += bck_cls.item()
        ldic['ptsmt'][1] += 1
        ldic['ptbck'][1] += 1

        tcls_losses = [smt_cls, bck_cls]

        # perform phase2 training as finetune_in is either 0 or 1
        for tcls_loss in tcls_losses:
            tcls_loss.backward()
        optim.step()

        # print training information
        if step != 0 and step % opt.pevery == 0:
            # for log messages
            avg_smt_cls = ldic['ptsmt'][0]/ldic['ptsmt'][1]
            avg_bck_cls = ldic['ptbck'][0]/ldic['ptbck'][1]

            ldic['ptsmt'] = [0., 0.]
            ldic['ptbck'] = [0., 0.]

            ldic['tsmt'].append(avg_smt_cls)
            ldic['tbck'].append(avg_bck_cls)

            lrs = []
            for params in optim.param_groups():
                for param in params:
                    lrs.append(param['lr'])  # learning rate.

            # log messages
            log_msg = "Model training information\n"
            log_msg += "semantic cls loss: %.5f, " % avg_smt_cls
            log_msg += "background cls loss: %.5f" % avg_bck_cls
            log_msg += "\nsemantic learning rate: %.6f, " % lrs[0]
            log_msg += "background learning rate: %.6f, " % lrs[1]
            logging.info(log_msg)

        # validation
        if step != 0 and step % opt.vevery == 0:
            models = models.eval()
            with torch.no_grad():
                vstep_in = 0
                vlosses = np.array([0., 0.])
                vsmt_correct, vbck_correct = 0, 0
                for vlocal_batch, vlocal_zbin in db['val_in']:
                    # classification
                    vlocal_batch = vlocal_batch.cuda()
                    vlocal_zbin = vlocal_zbin.cuda()

                    vsmt_probs, vbck_probs = models(vlocal_batch)

                    smt_vcls = opt.cls_criterion(vsmt_probs, vlocal_zbin)
                    bck_vcls = opt.cls_criterion(vbck_probs, vlocal_zbin)

                    vlosses[0] += smt_vcls
                    vlosses[1] += bck_vcls

                    ldic['pvsmt'] += smt_vcls
                    ldic['pvbck'] += bck_vcls

                    # get the index of the maximum probability
                    vsmt_pred = vsmt_probs.data.max(1, keepdim=True)[1]
                    vbck_pred = vbck_probs.data.max(1, keepdim=True)[1]
                    vsmt_correct += vsmt_pred.eq(
                        vlocal_zbin.data.view_as(vsmt_pred)).cpu().sum()
                    vbck_correct += vbck_pred.eq(
                        vlocal_zbin.data.view_as(vbck_pred)).cpu().sum()

                    vstep_in += 1

                ldic['pvsmt'] = 0.
                ldic['pvbck'] = 0.

                avg_vlosses = vlosses/vstep_in
                cls_ckpt_cond = avg_vlosses[0] < bcv[0] or \
                    avg_vlosses[1] < bcv[1]

                if cls_ckpt_cond:
                    if avg_vlosses[0] < bcv[0]:
                        bcv[0] = avg_vlosses[0]
                    if avg_vlosses[1] < bcv[1]:
                        bcv[1] = avg_vlosses[1]

                    msg = "Validation cls being smaller than previous minimum"
                    msg += ", checkpoint save condition is met."
                    logging.info(msg)

                    checkpoint = Checkpoint(
                        step, epoch, models, optim, opt=opt)
                    checkpoint.save()

                # for validation log message
                vdat_len = len(db['val_in'].dataset)
                universal_acc = float(vsmt_correct)/vdat_len
                specific_acc = float(vbck_correct)/vdat_len

                log_msg = \
                    "Validation semantic accuracy: %i/%i (%.6f)\n" % \
                    (vsmt_correct, vdat_len, universal_acc)
                log_msg += \
                    "Validation background accuracy: %i/%i (%.6f)\n" % \
                    (vbck_correct, vdat_len, specific_acc)
                log_msg += "current validation semantic cls: %.5f, " % \
                    avg_vlosses[0]
                log_msg += "minimum semantic cls: %.5f\n" % bcv[0]
                log_msg += "current validation background cls: %.5f, " % \
                    avg_vlosses[1]
                log_msg += "minimum background cls: %.5f\n" % bcv[1]
                logging.info(log_msg)

            models = models.train()
        step += 1

    if epoch >= opt.lr_decay_epoch:
        optim.update(avg_vlosses, epoch)

    return bcv


def lkhd_train(db, models, optim, opt):
    logging.info(">> Likelihood train")

    step = 0
    stime = time.time()
    bcv = [np.inf, np.inf]
    for epoch in range(opt.max_training_epoch):
        bcv = lkhd_train_epoch(models, optim, db, step, epoch, bcv, opt)

        log_msg = "Finished epoch %s/%s" % (epoch, opt.max_training_epoch)
        logging.info(log_msg)

    etime = time.time()  # end time
    dur = etime - stime  # training time
    logging.info("Training is done. Took %.3fh" % (dur/3600.))


def compute_loglkhd(probs, target):
    logp = torch.log(probs)

    nll = torch.nn.NLLLoss(reduction='none')
    loglkhd = -nll(logp, target)

    return loglkhd


def lkhd_evaluate(db, models, opt):
    logging.info(">> Likelihood evaluate")

    # model setting
    models, _, _ = set_loaded_model(models, opt)
    models = models.cuda().eval()

    eval_set_in = db['eval_in'].dataset
    zspec_in = eval_set_in.z

    smt_ncorrect = 0  # redshift classification
    bck_ncorrect = 0
    smt_lls_in, bck_lls_in = [], []
    with torch.no_grad():
        for bepoch, (local_batch_in, local_zbin_in) in \
                enumerate(db['eval_in']):
            binc = eval_set_in.binc.cuda()
            local_batch_in = local_batch_in.cuda()
            local_zbin_in = local_zbin_in.cuda()

            # input into model
            smt_probs_in, bck_probs_in = models(local_batch_in)

            smt_ll_in = compute_loglkhd(smt_probs_in, local_zbin_in)
            bck_ll_in = compute_loglkhd(bck_probs_in, local_zbin_in)

            smt_lls_in.append(smt_ll_in.cpu().detach().numpy().ravel())
            bck_lls_in.append(bck_ll_in.cpu().detach().numpy().ravel())

            # get the index of the maximum log-probability
            smt_pred_in = \
                smt_probs_in.data.max(1, keepdim=True)[1]
            bck_pred_in = \
                bck_probs_in.data.max(1, keepdim=True)[1]
            smt_correct = smt_pred_in.eq(
                local_zbin_in.data.view_as(smt_pred_in))
            bck_correct = bck_pred_in.eq(
                local_zbin_in.data.view_as(bck_pred_in))
            smt_ncorrect += smt_correct.cpu().sum()
            bck_ncorrect += bck_correct.cpu().sum()

            smt_zphot_in = torch.sum(
                smt_probs_in*binc, dim=1).view(-1)
            bck_zphot_in = torch.sum(
                bck_probs_in*binc, dim=1).view(-1)

            if not bepoch:
                # [batch_size, 1]
                smt_zphot_in_stacked = smt_zphot_in.cpu().detach()
                bck_zphot_in_stacked = bck_zphot_in.cpu().detach()
            else:
                # [batch_size*(bepoch+1), 1]
                smt_zphot_in_stacked = torch.cat(
                    (smt_zphot_in_stacked, smt_zphot_in.cpu().detach()), 0)
                bck_zphot_in_stacked = torch.cat(
                    (bck_zphot_in_stacked, bck_zphot_in.cpu().detach()), 0)

        smt_lls_in = np.hstack(smt_lls_in)
        bck_lls_in = np.hstack(bck_lls_in)
        lls_label_in = np.vstack((smt_lls_in,
                                  bck_lls_in,
                                  np.ones(len(smt_lls_in))))

        deval_len = len(eval_set_in)
        smt_correct = float(smt_ncorrect)/deval_len
        bck_correct = float(bck_ncorrect)/deval_len
        log_msg = "semantic eval set accuracy: %i/%i (%.6f), " % \
            (smt_ncorrect, deval_len, smt_correct)
        log_msg += "background eval set accuracy: %i/%i (%.6f)\n" % \
            (bck_ncorrect, deval_len, bck_correct)
        logging.info(log_msg)

        smt_outputs_in = smt_zphot_in_stacked.numpy()
        bck_outputs_in = bck_zphot_in_stacked.numpy()
        zphots_in = np.vstack((smt_outputs_in, bck_outputs_in))

        smt_lls_id = []
        bck_lls_id = []
        smt_lls_ood = []
        bck_lls_ood = []
        for local_batch_ul, local_y_ul, local_zbin_ul in db['eval_lo']:
            local_batch_ul = local_batch_ul.cuda()
            local_zbin_ul = local_zbin_ul.cuda()

            smt_probs_ul, bck_probs_ul = models(local_batch_ul)

            smt_ll_ul = compute_loglkhd(smt_probs_ul, local_zbin_ul)
            bck_ll_ul = compute_loglkhd(bck_probs_ul, local_zbin_ul)

            id_mask = local_y_ul == 1
            ood_mask = local_y_ul != 1

            smt_lls_id.append(
                smt_ll_ul.cpu().detach().view(-1)[id_mask].numpy())
            smt_lls_ood.append(
                smt_ll_ul.cpu().detach().view(-1)[ood_mask].numpy())
            bck_lls_id.append(
                bck_ll_ul.cpu().detach().view(-1)[id_mask].numpy())
            bck_lls_ood.append(
                bck_ll_ul.cpu().detach().view(-1)[ood_mask].numpy())

        smt_lls_id = np.hstack(smt_lls_id)
        bck_lls_id = np.hstack(bck_lls_id)
        lls_label_id = np.vstack((smt_lls_id,
                                  bck_lls_id,
                                  np.ones(len(smt_lls_id))))
        smt_lls_ood = np.hstack(smt_lls_ood)
        bck_lls_ood = np.hstack(bck_lls_ood)
        lls_label_ood = np.vstack((smt_lls_ood,
                                   bck_lls_ood,
                                   np.zeros(len(smt_lls_ood))))

    ll_label = [lls_label_in.T, lls_label_id.T, lls_label_ood.T]

    save_outputs = [zphots_in.T]+ll_label
    save_results(save_outputs, opt)

    return zspec_in, zphots_in, ll_label


def set_loaded_model(models, opt, optim=None):
    resume_checkpoint = Checkpoint.load(
        models, optim=optim, opt=opt)
    models = resume_checkpoint.models
    epoch = resume_checkpoint.epoch
    step = resume_checkpoint.step

    if optim is not None:
        optim = resume_checkpoint.optim

    return models, optim, [epoch, step]


def save_results(outputs, opt, fn=None):
    if not os.path.exists(opt.quant_fd):
        os.makedirs(opt.quant_fd)
    if fn is None:
        out_fn = os.path.join(opt.quant_fd, opt.out_fn)
    else:
        out_fn = os.path.join(opt.quant_fd, fn)

    np.save(out_fn, outputs)
    logging.info("Outputs are saved at %s" % out_fn)


def save_losses(opt, ldic=None, epoch=None, delete=False):
    if delete:
        if os.path.exists(opt.loss_fd):
            shutil.rmtree(opt.loss_fd)
    else:
        if not os.path.exists(opt.loss_fd):
            os.makedirs(opt.loss_fd)

        cls_train_fn = os.path.join(opt.loss_fd, 'train_cls_loss_%s' % epoch)
        dcp_train_fn = os.path.join(opt.loss_fd, 'train_dcp_loss_%s' % epoch)
        dcp_val_fn = os.path.join(opt.loss_fd, 'val_dcp_loss_%s' % epoch)

        cls_train_loss = np.vstack((ldic['tuniv'], ldic['tspec'])).T

        f = open(cls_train_fn, 'w')
        np.savetxt(f, cls_train_loss)
        f.close()
        logging.info("Train cls loss is saved at %s" % cls_train_fn)

        if len(ldic['tdcp_in']) and len(ldic['tdcp_ul']):
            dcp_train_loss = np.vstack((ldic['tdcp_in'], ldic['tdcp_ul'])).T
            f = open(dcp_train_fn, 'w')
            np.savetxt(f, dcp_train_loss)
            f.close()

            logging.info("Train dcp loss is saved at %s" % dcp_train_fn)

        if len(ldic['vdcp_in']) and len(ldic['vdcp_ul']):
            dcp_val_loss = np.vstack((ldic['vdcp_in'], ldic['vdcp_ul'])).T
            f = open(dcp_val_fn, 'w')
            np.savetxt(f, dcp_val_loss)
            f.close()

            logging.info("Validation dcp loss is saved at %s" % dcp_val_fn)


def print_model_grad_for_debug(model):
    for m in model.modules():
        for p in m.named_parameters():
            if 'widening' in p[0] and 'weight' in p[0]:
                print(p[0], p[1].grad.min(), p[1].grad.max())
