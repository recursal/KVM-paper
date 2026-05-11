import torch


def orthogonal_(weight, gain):
    with torch.no_grad():
        weight.copy_(
            torch.nn.init.orthogonal_(torch.empty_like(weight, dtype=torch.float), gain)
        )


def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = (shape[0] / shape[1]) ** 0.5 if shape[0] > shape[1] else 1
            orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = (shape[1] / shape[2]) ** 0.5 if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                orthogonal_(x[i], gain=gain * scale)
        else:
            assert False
        return x
