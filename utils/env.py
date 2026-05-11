import os

_configure_torch_cache_env_complete = False
def setup_env(verbose: bool = True) -> None:
    global _configure_torch_cache_env_complete
    if _configure_torch_cache_env_complete:
        return
    _configure_torch_cache_env_complete = True
    rank = int(os.environ.get("RANK", 0))
    # set these BEFORE importing torch / triton just in case
    if os.environ.get("TRITON_CACHE_DIR") is None:
        if verbose and rank == 0:
            print("setting TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = f"/local/.triton/cache"
    if os.environ.get("TORCHINDUCTOR_DIR") is None:
        if verbose and rank == 0:
            print("setting TORCHINDUCTOR_DIR")
        os.environ["TORCHINDUCTOR_DIR"] = f"/local/.torchinductor"
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") is None:
        if verbose and rank == 0:
            print("setting TORCHINDUCTOR_CACHE_DIR")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/local/.torchinductor/cache"

setup_env(verbose=False)
