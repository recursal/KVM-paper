from torch.distributed import get_rank


def print0(*args, **kwargs):
    try:
        rank = get_rank()
    except:
        rank = -1
    if rank <= 0:
        print(*args, **kwargs)
