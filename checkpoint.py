import os
import time
import logging
import shutil

import torch

model_state = 'model_state_%s.pt'
trainer_state = 'trainer_state_%s.pt'


class Checkpoint():

    def __init__(self, step, epoch, models, optim=None, path=None, opt=None):
        self.step = step
        self.epoch = epoch
        self.models = models

        self.optim = optim
        self._path = path
        self.opt = opt

        self.logger = logging.getLogger(__name__)

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    @classmethod
    def load(cls, models, optim=None, opt=None):
        logger = logging.getLogger(__name__)

        all_times = sorted(os.listdir(opt.ckpt_fd), reverse=True)
        fchckpt = os.path.join(
            opt.ckpt_fd, all_times[0])

        logger.info("load model state from %s" %
                    os.path.join(fchckpt, model_state % opt.load_key))
        resume_model = torch.load(
            os.path.join(fchckpt, model_state % opt.load_key),
            map_location=opt.device)
        logger.info("load trainer state from %s" %
                    os.path.join(fchckpt, trainer_state % opt.load_key))
        resume_checkpoint = torch.load(
            os.path.join(fchckpt, trainer_state % opt.load_key),
            map_location=opt.device)

        models[0].load_state_dict(resume_model['universal_network_state'])
        models[1].load_state_dict(resume_model['specific_network_state'])
        if optim is not None:
            optim.load_state_dict(resume_checkpoint['optimizer'])

        return Checkpoint(step=resume_checkpoint['step'],
                          epoch=resume_checkpoint['epoch'],
                          models=models,
                          optim=optim,
                          path=opt.ckpt_fd)

    def save(self, ftune, ptune):
        if not ftune and not ptune:
            key = 'pre_trained'
            outdated = os.listdir(self.opt.ckpt_fd)
            for od in outdated:
                self.logger.info("Remove outdated checkpoint %s" % od)
                shutil.rmtree(os.path.join(self.opt.ckpt_fd, od))

            date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        elif ftune and not ptune:
            # overwrite
            key = 'TS2'
            date_time = sorted(os.listdir(self.opt.ckpt_fd), reverse=True)[0]
        elif not ftune and ptune:
            # overwrite
            key = 'TS3'
            dates = os.listdir(self.opt.ckpt_fd)
            if not len(dates):
                date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            else:
                date_time = sorted(dates, reverse=True)[0]
        else:
            raise NotImplementedError

        path = os.path.join(
            self.opt.ckpt_fd, date_time)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {'epoch': self.epoch,
             'step': self.step,
             'optimizer': self.optim.state_dict()},
            os.path.join(path, trainer_state % key))
        torch.save(
            {'universal_network_state': self.models[0].state_dict(),
             'specific_network_state': self.models[1].state_dict()},
            os.path.join(path, model_state % key))

        log_msg = "Checkpoint is saved at %s\n" % path
        self.logger.info(log_msg)
