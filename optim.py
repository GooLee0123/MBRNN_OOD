import itertools

import torch


class Optimizer():

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, max_grad_norm=-1):
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def step(self):
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss, epoch)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, loaded_dict):
        self.optimizer.load_state_dict(loaded_dict)

    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = \
                                subparam._grad.data.to(device)


class Optimizers():

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optims, max_grad_norm=-1):
        self.optimizers = optims
        self.schedulers = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, schedulers):
        self.schedulers = []
        for scheduler in schedulers:
            self.schedulers.append(scheduler)

    def step(self):
        for optimizer in self.optimizers:
            if self.max_grad_norm > 0:
                params = itertools.chain.from_iterable(
                    [group['params'] for group in optimizer.param_groups])
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            optimizer.step()

    def update(self, losses, epoch):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler is None:
                pass
            elif isinstance(scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(losses[i], epoch)
            else:
                scheduler.step()

    def state_dict(self):
        state_dicts = []
        for optimizer in self.optimizers:
            state_dicts.append(optimizer.state_dict())
        return state_dicts

    def load_state_dict(self, loaded_dicts):
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(loaded_dicts[i])

    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.append(optimizer.param_groups)
        return param_groups

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
