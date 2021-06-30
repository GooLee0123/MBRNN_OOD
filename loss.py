#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class AnchorLoss(nn.Module):
    r"""Anchor Loss: modulates the standard cross entropy based on
            the prediction difficulty.
            Loss(x, y) = - y * (1 - x + p_pos)^gamma_pos * \log(x)
                        - (1 - y) * (1 + x - p_neg)^gamma_neg * \log(1-x)

            The losses are summed over class and averaged across observations
            for each minibatch.


        Args:
            gamma(float, optional): gamma > 0; reduces the relative loss
            for well-classiﬁed examples,
                                    putting more focus on hard, misclassiﬁed
                                    examples
            slack(float, optional): a margin variable to penalize the output
            variables which are close to
                                    true positive prediction score
            warm_up(bool, optional): if ``True``, the loss is replaced to
            cross entropy for the first 5 epochs,
                                     and additional epoch variable which
                                     indicates the current epoch is needed
            anchor(string, optional): specifies the anchor probability type:
                                      ``pos``: modulate target class loss
                                      ``neg``: modulate background class loss
        Shape:
            - Input: (N, C) where C is the number of classes
            - Target: (N) where each value is the class label of each sample
            - Epoch: int, optional variable when using warm_up
            - Output: scalar

    """
    def __init__(self, gamma=0.5, slack=0.05, anchor='neg', sigma=2.):
        super(AnchorLoss, self).__init__()

        assert anchor in ['neg', 'pos'], \
            "Anchor type should be either ``neg`` or ``pos``"

        self.gamma = gamma
        self.slack = slack
        self.warm_up = warm_up
        self.anchor = anchor
        self.sigma = sigma
        self.sig = nn.Sigmoid()
        self.EPS = 1e-12  # for preventing nan or inf

        if anchor == 'pos':
            self.gamma_pos = gamma
            self.gamma_neg = 0
        elif anchor == 'neg':
            self.gamma_pos = 0
            self.gamma_neg = gamma

    def forward(self, input, target, epoch=None):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        if self.warm_up and epoch is None:
            raise AssertionError(
                "If warm_up is set to ``True``, "
                + "current epoch number is required")

        if self.warm_up and epoch < 5:
            loss = self.ce(input, target)
            return loss

        target = target.view(-1, 1)
        pt = input
        logpt_pos = torch.log(input + self.EPS)
        logpt_neg = torch.log(1 - pt + self.EPS)  # log(1-q)

        N = input.size(0)
        C = input.size(1)

        class_mask = input.data.new(N, C).fill_(0)
        class_mask.scatter_(1, target.data, 1.)
        class_mask = class_mask.float()

        pt_pos = pt.gather(1, target).view(-1, 1)
        pt_neg = pt * (1-class_mask)
        pt_neg = pt_neg.max(dim=1)[0].view(-1, 1)
        pt_neg = (pt_neg + self.slack).clamp(max=1).detach()
        pt_pos = (pt_pos - self.slack).clamp(min=0).detach()

        scaling_pos = -1 * (1 - pt + pt_neg).pow(self.gamma_pos)
        loss_pos = scaling_pos * logpt_pos
        scaling_neg = -1 * (1 + pt - pt_pos).pow(self.gamma_neg)
        loss_neg = scaling_neg * logpt_neg

        loss = class_mask * loss_pos + (1 - class_mask) * loss_neg
        loss = loss.sum(1)

        return loss.mean()


class DiscrepancyLoss(nn.Module):
    r"""Discrepancy Loss: estimates entropy difference between the
            given two probability distributions.
        Loss(p, q) = H(p) - H(q), where H() is entropy.


        Args:
            margin(float, optional): maximum value of the loss for
                                     preventing overfitting.
            mean(bool, optional): if ``True``, return mean of losses
        Shape:
            - universal probs: (N, C) where C is the number of classes
            - specific probs: (N, C) where C is the number of classes
    """
    def __init__(self, opt, margin=3.0, mean=False):
        super(DiscrepancyLoss, self).__init__()

        self.device = opt.device
        self.margin = margin
        self.mean = mean

    def forward(self, universal_probs, specific_probs):
        entropy1 = Categorical(probs=universal_probs).entropy()
        entropy2 = Categorical(probs=specific_probs).entropy()

        discrepancy = entropy1-entropy2

        if self.mean:
            loss = self.margin - discrepancy.mean()
            loss = max(torch.tensor(0.).to(self.device), loss)
        else:
            _eps = 1e-5
            _len = universal_probs.size(1)
            _uni = torch.zeros(_len)+1./_len
            _max = Categorical(probs=_uni).entropy().item()
            _min = 0 + _eps
            loss = torch.clamp(discrepancy, _min, _max)
            # loss = discrepancy

        return loss
