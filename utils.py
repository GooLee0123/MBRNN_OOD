import shutil
import logging
import os
import time
import copy

import numpy as np
import torch

from checkpoint import Checkpoint
from optim import Optimizer
from collections import namedtuple


def loss_containers(opt):
    ldic = {}

    # t: train, v: val, p: print
    ldic['ptHE'] = [0., 0.]
    ldic['pvHE'] = 0.

    ldic['ptLE'] = [0., 0.]
    ldic['pvLE'] = 0.

    ldic['tdcp_id'] = []
    ldic['vdcp_id'] = []
    ldic['ptdcp_id'] = [0., 0.]
    ldic['pvdcp_id'] = 0.

    ldic['tdcp_ul'] = []
    ldic['vdcp_ul'] = []
    ldic['ptdcp_ul'] = [0., 0.]
    ldic['pvdcp_ul'] = 0.

    return ldic


def unsup_train_epoch(models, optim, db, step, epoch, do_TS2,
                      do_TS3, TS2tol, best_vcls, best_vdcp_diff, opt):
    TS2_idx = 0
    ldic = loss_containers(opt)

    # train_switch = False
    models = models.cuda()
    models = models.train()
    for Be, (local_batch_id, local_zbin_id) in enumerate(db['train_id']):
        local_batch_id = local_batch_id.cuda()
        local_zbin_id = local_zbin_id.cuda()

        optim.zero_grad()

        if do_TS2:
            if TS2_idx < 2:
                local_batch_ul, _ = next(db['train_ul_cycle'])
                local_batch_ul = local_batch_ul.cuda()

                local_batch = torch.cat((local_batch_id, local_batch_ul), 0)

                nin = local_batch_id.size(0)
            else:
                local_batch = local_batch_id
        else:
            local_batch = local_batch_id

        HEprobs, LEprobs = models(local_batch)

        if do_TS2:
            if TS2_idx < 2:
                HEprobs_id = HEprobs[:nin]
                HEprobs_ul = HEprobs[nin:]

                LEprobs_id = LEprobs[:nin]
                LEprobs_ul = LEprobs[nin:]
            else:
                HEprobs_id = HEprobs
                LEprobs_id = LEprobs
        else:
            HEprobs_id = HEprobs
            LEprobs_id = LEprobs

        HEloss_anch = opt.cls_criterion(HEprobs_id, local_zbin_id)
        LEloss_anch = opt.cls_criterion(LEprobs_id, local_zbin_id)

        ldic['ptHE'][0] += HEloss_anch.item()
        ldic['ptLE'][0] += LEloss_anch.item()
        ldic['ptHE'][1] += 1
        ldic['ptLE'][1] += 1

        tcls_loss = HEloss_anch + LEloss_anch

        if do_TS2:
            if TS2_idx < 2:
                tdcp_id = opt.dcp_criterion(HEprobs_id, LEprobs_id)
                tdcp_ul = opt.dcp_criterion(HEprobs_ul, LEprobs_ul)
                tcls_loss += opt.dcp_weight*tdcp_ul

                ldic['ptdcp_id'][0] += tdcp_id.item()
                ldic['ptdcp_ul'][0] += tdcp_ul.item()
                ldic['ptdcp_id'][1] += 1
                ldic['ptdcp_ul'][1] += 1

                TS2_idx += 1
            else:
                TS2_idx = 0

        tcls_loss.backward()
        optim.step()

        if step != 0 and step % opt.pevery == 0:
            # for log messages
            avg_HE_cls = ldic['ptHE'][0]/ldic['ptHE'][1]
            avg_LE_cls = ldic['ptLE'][0]/ldic['ptLE'][1]

            ldic['ptHE'] = [0., 0.]
            ldic['ptLE'] = [0., 0.]

            if do_TS2:
                avg_dcp_id = ldic['ptdcp_id'][0]/ldic['ptdcp_id'][1]
                avg_dcp_ul = ldic['ptdcp_ul'][0]/ldic['ptdcp_ul'][1]

                ldic['ptdcp_id'] = [0., 0.]
                ldic['ptdcp_ul'] = [0., 0.]

                ldic['tdcp_id'].append(avg_dcp_id)
                ldic['tdcp_ul'].append(avg_dcp_ul)

                for g in optim.param_groups():
                    g['lr'] = opt.TS2_lr

            for param in optim.param_groups():
                lr = param['lr']  # learning rate.

            # log messages
            log_msg = "(TS2) " if do_TS2 else "(TS3) "\
                if do_TS3 else ""
            log_msg += "ANCHOR loss (HE): %.3f, " % avg_HE_cls
            log_msg += "ANCHOR loss (LE): %.3f" % avg_LE_cls
            if do_TS2:
                log_msg += ", dcp loss (ID): %.3f" % avg_dcp_id
                log_msg += ", dcp loss (UL): %.3f" % avg_dcp_ul

            log_msg += ", learning rate: %.5f" % lr
            logging.info(log_msg)

        # validation
        if step != 0 and step % opt.vevery == 0:
            models = models.eval()
            with torch.no_grad():
                vstep_id, vcls = 0, 0
                for vlocal_batch_id, vlocal_zbin_id in db['val_id']:
                    # classification
                    vlocal_batch_id = vlocal_batch_id.cuda()
                    vlocal_zbin_id = vlocal_zbin_id.cuda()

                    vHE_probs_id, vLE_probs_id = models(vlocal_batch_id)

                    HEloss_anch = opt.cls_criterion(vHE_probs_id, vlocal_zbin_id)
                    LEloss_anch = opt.cls_criterion(vLE_probs_id, vlocal_zbin_id)

                    vcls += HEloss_anch + LEloss_anch

                    # discrepancy
                    vdcp_id = opt.dcp_criterion(vHE_probs_id, vLE_probs_id)

                    ldic['pvHE'] += HEloss_anch
                    ldic['pvLE'] += LEloss_anch

                    ldic['pvdcp_id'] += vdcp_id

                    vstep_id += 1

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

                    vHE_probs_ul, vLE_probs_ul = models(vlocal_batch_ul)

                    vdcp_loss_ul = opt.dcp_criterion(vHE_probs_ul, vLE_probs_ul)

                    ldic['pvdcp_ul'] += vdcp_loss_ul
                    vstep_ul += 1

                avg_vdcp_id = ldic['pvdcp_id']/vstep_id
                avg_vdcp_ul = ldic['pvdcp_ul']/vstep_ul

                ldic['pvHE'] = 0.
                ldic['pvLE'] = 0.
                ldic['pvdcp_id'] = 0.
                ldic['pvdcp_ul'] = 0.

                ldic['vdcp_id'].append(avg_vdcp_id)
                ldic['vdcp_ul'].append(avg_vdcp_ul)

                avg_vloss = vcls/vstep_id
                cls_ckpt_cond = avg_vloss < best_vcls

                vdcp_diff = avg_vdcp_id-avg_vdcp_ul
                dcp_ckpt_cond = do_TS2 and vdcp_diff > best_vdcp_diff
                if cls_ckpt_cond or dcp_ckpt_cond:
                    if cls_ckpt_cond:
                        best_vcls = avg_vloss
                    if dcp_ckpt_cond:
                        best_vdcp_diff = vdcp_diff

                    msg = "\ncheckpoint save condition is met."
                    logging.info(msg)

                    checkpoint = Checkpoint(step, epoch, models, optim, opt=opt)
                    checkpoint.save(do_TS2, do_TS3)
                if cls_ckpt_cond or dcp_ckpt_cond:
                    TS2tol = 0
                else:
                    TS2tol += 1

                if not do_TS3 and not do_TS2 and TS2tol > opt.ts2_patience:
                    logging.info("Start TS2")
                    do_TS2 = True
                    TS2tol = 0

                if do_TS2 and TS2tol > opt.ts3_patience:
                    logging.info("Stop TS2 and start TS3")
                    do_TS2 = False
                    do_TS3 = True

                # for validation log message
                vdat_len = len(db['val_id'].dataset)

                log_msg = "Current validation ANCHOR loss: %.5f, " % avg_vloss
                log_msg += "Current validation ID-DCP loss: %.5f, " % avg_vdcp_id
                log_msg += "UL-DCP loss: %.5f\n" % avg_vdcp_ul
                logging.info(log_msg)

            models = models.train()
        step += 1

    if epoch >= opt.lr_decay_epoch:
        optim.update(avg_vloss, epoch)

    return best_vcls, best_vdcp_diff, do_TS2, do_TS3,\
        TS2tol, ldic, step, optim, db


def unsup_train(db, models, optim, opt):
    logging.info(">> Unsupervised train")

    sepoch, step = [0, 0]
    stime = time.time()  # start time
    bcv = np.inf  # minimum validation loss placeholder
    bdv = 0  # maximum discrepancy difference between in and ul data
    do_TS2 = False
    do_TS3 = False
    TS2tol = 0
    for epoch in range(sepoch, opt.max_training_epoch):
        args = unsup_train_epoch(models, optim, db, step, epoch, do_TS2,
                                 do_TS3, TS2tol, bcv, bdv, opt)
        bcv, bdv, do_TS2, do_TS3, TS2tol, ldic, step, optim, db = args

        log_msg = "Finished epoch %s/%s" % (epoch, opt.max_training_epoch)
        logging.info(log_msg)

    etime = time.time()  # end time
    dur = etime - stime  # training time
    logging.info("Training is done. Took %.3fh" % (dur/3600.))


def get_placeholder(nrow, ncol):
    Placeholder = namedtuple('Placeholder', ['p1', 'p2', 'za1', 'za2', 'dcp'])

    ph = Placeholder(torch.empty(nrow, ncol).cuda(),
                     torch.empty(nrow, ncol).cuda(),
                     torch.empty(nrow).cuda(),
                     torch.empty(nrow).cuda(),
                     torch.empty(nrow).cuda())
    return ph


def unsup_evaluate(db, models, opt):
    logging.info(">> Unsupervised evaluate")

    # model setting
    models, _, _ = set_loaded_model(models, opt)

    models = models.eval().zero_grad().cuda()

    eval_set_id = db['eval_id'].dataset
    eval_set_lo = db['eval_lo'].dataset
    eval_set_ul = db['eval_ul'].dataset
    zLE_id = eval_set_id.z.numpy()
    zLE_lo = eval_set_lo.z.numpy()

    idlen = len(eval_set_id)
    lolen = len(eval_set_lo)
    ullen = len(eval_set_ul)

    dcp_id = []
    placeholder_id = get_placeholder(idlen, opt.ncls)
    placeholder_lo = get_placeholder(lolen, opt.ncls)
    placeholder_ul = get_placeholder(ullen, opt.ncls)

    fnames = ['output_id', 'output_lood', 'output_ul']
    dbs = [db['eval_id'], db['eval_lo'], db['eval_ul']]
    phs = [placeholder_id, placeholder_lo, placeholder_ul]
    with torch.no_grad():
        for i in range(3):
            for bepoch, (local_batch, local_zbin) in enumerate(dbs[i]):
                binc = eval_set_id.binc.cuda()
                local_batch = local_batch.cuda()
                local_zbin = local_zbin.cuda()

                # input into model
                HEprobs, LEprobs = models(local_batch)

                dcp_loss = opt.dcp_criterion(HEprobs, LEprobs)

                HEzphot = torch.sum(HEprobs*binc, dim=1).view(-1)
                LEzphot = torch.sum(LEprobs*binc, dim=1).view(-1)

                store_sidx = bepoch*opt.batch_size
                store_eidx = store_sidx+opt.batch_size

                phs[i].p1[store_sidx:store_eidx] = HEprobs
                phs[i].p2[store_sidx:store_eidx] = LEprobs
                phs[i].za1[store_sidx:store_eidx] = HEzphot
                phs[i].za2[store_sidx:store_eidx] = LEzphot
                phs[i].za2[store_sidx:store_eidx] = dcp_loss

            outputs = get_outputs(placeholder_id)
            save_results(outputs, opt, fnames[i])


def get_outputs(ph):
    return np.hstack([ph.p1.cpu().detach().numpy(),
                     ph.p2.cpu().detach().numpy(),
                     ph.za1.cpu().detach().numpy()[:, np.newaxis],
                     ph.za2.cpu().detach().numpy()[:, np.newaxis],
                     ph.dcp.cpu().detach().numpy()[:, np.newaxis]])


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
