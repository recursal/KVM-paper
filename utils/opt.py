from torch import Tensor


def set_label(label, module, mup_d_in_ratio=None):
    if isinstance(module, Tensor):
        module.label = label
        module.mup_d_in_ratio = mup_d_in_ratio
    else:
        for p in module.parameters():
            set_label(label, p, mup_d_in_ratio)
    return module
