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
            # if bn_freeze:
            #     for m in model.modules():
            #         if isinstance(m, nn.BatchNorm1d):
            #             m.eval()
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False

        return Models(self.models, self.opt,
                      phase=self.phase)

    def eval(self):
        self.phase = 'eval'
        for model in self.models:
            model.eval()
        return Models(self.models, self.opt,
                      phase=self.phase)

    def batch_train(self):
        for model in self.models:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.train()
                    # m.weight.requires_grad = True
                    # m.bias.requires_grad = True
        return Models(self.models, self.opt,
                      phase=self.phase)

    def batch_eval(self):
        for model in self.models:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    # m.weight.requires_grad = False
                    # m.bias.requires_grad = False
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

    def _noise(self, inputs, nu=0.1):
        noise = inputs.data.new(inputs.size()).normal_(0, nu)
        return inputs + noise

    def _perturb(self, x, nu=0.2):
        b, c = x.size()
        mask = torch.rand(b, c) < nu
        mask = mask.float().cuda()
        noise = torch.FloatTensor(x.size()).uniform_().cuda()
        perturbed_x = (1-mask)*x + mask*noise
        return perturbed_x

    def __call__(self, inputs):
        outs = []
        for i, model in enumerate(self.models):
            if self.phase == 'train' and self.method == 'lkhd' and i == 1:
                if self.noise == 'noise':
                    inputs = self._noise(inputs, self.opt.nu)
                elif self.noise == 'perturb':
                    inputs = self._perturb(inputs, self.opt.nu)
                else:
                    raise NotImplementedError()

            outs.append(model(inputs))
        return outs

    def __getitem__(self, item_number):
        return self.models[item_number]


class Surrogate_Network(nn.Module):
    def __init__(self, n_members, ncls=128, case='case1'):
        super(Surrogate_Network, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.case = case

        if case == 'case1':
            self.weights = np.array([1.0/n_members for _ in range(n_members)])
            self.weights = Parameter(
                torch.from_numpy(self.weights).view(-1, 1, 1),
                requires_grad=True)
        elif case == 'case2':
            self.weights = torch.abs(torch.randn(n_members, 1, ncls))
            self.weights = Parameter(self.weights, requires_grad=True)

    def forward(self, probs):
        probs = probs.clone()
        if self.case == 'case1':
            normed_weights = torch.abs(self.weights)/torch.sum(
                    torch.abs(self.weights), dim=0)
            weighted_probs = torch.sum(normed_weights*probs, dim=0)
        elif self.case == 'case2':
            total_sum = torch.tensordot(
                torch.abs(self.weights), probs, dims=([0, 2], [0, 2]))
            normed_weights = torch.abs(self.weights.permute(0, 2, 1))/total_sum
            normed_weights = normed_weights.permute(0, 2, 1)
            weighted_probs = torch.sum(normed_weights*probs, dim=0)

        return weighted_probs


class Ensemble_E3(nn.Module):
    '''
        Surrogate model for ensemble learning
    '''
    def __init__(self, n_members, ncls=128):
        super(Ensemble_E3, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.weights = torch.abs(torch.randn(n_members, ncls, 1))
        self.weights = Parameter(self.weights, requires_grad=True)

    def forward(self, probs):
        total_sum = torch.tensordot(
            torch.abs(self.weights), probs, dims=([0, 1], [0, 1]))
        normed_weights = torch.abs(self.weights)/total_sum
        weighted_probs = torch.sum(normed_weights*probs, dim=0)

        return weighted_probs
