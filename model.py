import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MBRNN(nn.Module):

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, opt):
        super(MBRNN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ncls_ = opt.ncls
        self.batch_size = opt.batch_size
        self.classes_ = np.arange(self.ncls_)

        self.ninp = opt.ninp
        self.ncls = opt.ncls
        self.widening_layer = opt.widening_layer
        self.narrowing_layer = opt.narrowing_layer

        self.widening, self.narrowing = self.setup()

    def setup(self):
        widening_layer = list(map(int, self.widening_layer.split(',')))
        narrowing_layer = list(map(int, self.narrowing_layer.split(',')))

        widening_layer = [int(self.ninp)]+widening_layer
        arr = np.arange(len(widening_layer)-1)
        itr = np.column_stack((arr, arr, arr)).flatten()

        widening = [
            nn.Linear(widening_layer[j], widening_layer[j+1], bias=True)
            if i % 3 == 0 else nn.BatchNorm1d(widening_layer[j+1])
            if i % 3 == 1 else nn.Softplus()
            for i, j in enumerate(itr)]

        narrowing_layer = [widening_layer[-1]]+narrowing_layer+[self.ncls]
        carr = np.arange(len(narrowing_layer)-1)
        citr = np.column_stack((carr, carr, carr)).flatten()

        narrowing = [
            nn.Linear(narrowing_layer[j], narrowing_layer[j+1], bias=True)
            if i % 3 == 0 else nn.BatchNorm1d(narrowing_layer[j+1])
            if i % 3 == 1 else nn.Softplus()
            for i, j in enumerate(citr)]
        narrowing = narrowing[:-2]

        network_widening = nn.Sequential(*widening)
        network_narrowing = nn.Sequential(*narrowing)

        network_widening.apply(self._init_weights)
        network_narrowing.apply(self._init_weights)

        return network_widening, network_narrowing

    def forward(self, x):
        _x = self.widening(x)
        out = self.narrowing(_x)
        out = F.softmax(out, dim=1)

        return out


class Models():

    def __init__(self, models, opt, phase=None):
        self.models = models
        self.phase = phase

        self.method = opt.method
        self.noise = opt.noise
        self.opt = opt

        self.logger = logging.getLogger(__name__)

    def cuda(self):
        for model in self.models:
            model.cuda()
        return Models(self.models, self.opt,
                      phase=self.phase)

    def train(self):
        self.phase = 'train'
        for model in self.models:
            model.train()
        return Models(self.models, self.opt,
                      phase=self.phase)

    def eval(self):
        self.phase = 'eval'
        for model in self.models:
            model.eval()
        return Models(self.models, self.opt,
                      phase=self.phase)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()
        return Models(self.models, self.opt,
                      phase=self.phase)

    def apply(self, func):
        for model in self.models:
            model.apply(func)
        return Models(self.models, self.opt,
                      phase=self.phase)

    def __call__(self, inputs):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(inputs))
        return outs

    def __getitem__(self, item_number):
        return self.models[item_number]

