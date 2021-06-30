import os
import itertools
import logging

from option_parse import Parser

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

import model as Network
import PS1
import utils

from loss import AnchorLoss, DiscrepancyLoss
from optim import Optimizer, Optimizers
from ensemble import equal_bin_ensemble
from result_analysis import lkhd_analysis, unsup_analysis

torch.multiprocessing.set_sharing_strategy('file_system')


LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)


def prepare_optim(models, opt):
    if opt.optim.upper() == 'ADAM':
        optim = torch.optim.Adam(
                    list(models[0].parameters()) +
                    list(models[1].parameters()),
                    lr=0.0008,
                    betas=(0.5, 0.999),
                    weight_decay=5e-5)
    elif opt.optim.upper() == 'SGD':
        optim = torch.optim.SGD(
                    list(models[0].parameters()) +
                    list(models[1].parameters()),
                    lr=0.1,
                    weight_decay=5e-5)
    else:
        raise NotImplementedError()

    # setting optimizer
    optimizer = Optimizer(
        optim,
        max_grad_norm=3.)
    # setting scheduler of optimizer for learning rate decay.
    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,
        patience=10,
        factor=0.5,
        min_lr=0.000001)
    optimizer.set_scheduler(scheduler)

    return optimizer


def dataset_config(batch_size):
    dparams = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0}
    evdparams = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 0}

    return dparams, evdparams


def prepare_db(opt):
    db = {}
    dparams, evdparams = dataset_config(opt.batch_size)

    if opt.train:
        train_set_id = PS1.Dataset('train', opt)
        val_set_id = PS1.Dataset('val', opt)
        if opt.tr_ul_prefix == 'LOOD':
            train_set_ul = PS1.Dataset('train', opt, dopt='LOOD')
            val_set_ul = PS1.Dataset('val', opt, dopt='LOOD')
        else:
            train_set_ul = PS1.Dataset('train', opt, dopt='UL')
            val_set_ul = PS1.Dataset('val', opt, dopt='UL')

        train_loader_id = torch.utils.data.DataLoader(train_set_id, **dparams)
        train_loader_ul = torch.utils.data.DataLoader(train_set_ul, **dparams)
        val_loader_id = torch.utils.data.DataLoader(val_set_id, **evdparams)
        val_loader_ul = torch.utils.data.DataLoader(val_set_ul, **evdparams)

        db['train_id'] = train_loader_id
        db['train_ul_cycle'] = itertools.cycle(train_loader_ul)
        db['val_id'] = val_loader_id
        db['val_ul'] = val_loader_ul
    else:
        val_set_id = PS1.Dataset('val', opt)
        eval_set_id = PS1.Dataset('eval', opt)
        eval_set_ul = PS1.Dataset('eval', opt, dopt='UL')
        eval_set_lo = PS1.Dataset('eval', opt, dopt='LOOD')

        val_loader_id = torch.utils.data.DataLoader(val_set_id, **evdparams)
        eval_loader_id = torch.utils.data.DataLoader(eval_set_id, **evdparams)
        eval_loader_ul = torch.utils.data.DataLoader(eval_set_ul, **evdparams)
        eval_loader_lo = torch.utils.data.DataLoader(eval_set_lo, **evdparams)

        db['val_id'] = val_loader_id
        db['eval_id'] = eval_loader_id
        db['eval_ul'] = eval_loader_ul
        db['eval_lo'] = eval_loader_lo

    return db


def prepare_loss(opt):
    mean = True if opt.train else False
    opt.cls_criterion = AnchorLoss(gamma=opt.gamma)
    # opt.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    opt.dcp_criterion = DiscrepancyLoss(margin=opt.dcp_margin,
                                        mean=mean)

    return opt


def prepare_models(opt):
    opt.ninp = 17
    model1 = Network.MBRNN(opt)
    model2 = Network.MBRNN(opt)

    models = (model1, model2)
    models = Network.Models(models, opt)

    logging.info(model1)
    logging.info(model2)

    return models


def main():
    opt = Parser()

    torch.cuda.set_device(opt.gpuid)
    opt.device = torch.device('cuda:%s' % opt.gpuid)

    if not torch.cuda.is_available():
        raise NotImplementedError("CAN'T RUN WITHOUT GPU")

    db = prepare_db(opt)
    opt = prepare_loss(opt)
    models = prepare_models(opt)
    optim = prepare_optim(models, opt)

    if opt.train:
        utils.unsup_train(db, models, optim, opt)
    else:
        utils.unsup_evaluate(db, models, opt)


if __name__ == '__main__':
    main()
