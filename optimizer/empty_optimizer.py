import torch


class EmptyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        pass

    def step(self):
        pass

    @property
    def param_groups(self):
        return []

    def state_dict(self):
        return {}
