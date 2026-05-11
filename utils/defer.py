def defer(inner_decorator):
    """
    Wraps any decorator so it becomes a no-op at definition time,
    but can be applied later via apply_deferred(instance).

    Usage:
        @defer(torch.compile, mode="reduce-overhead")
        def forward(self, x): ...

        @defer(torch.jit.script)
        def forward(self, x): ...
    """

    def marker(fn):
        fn._deferred_decorator = inner_decorator
        return fn

    return marker


def apply_deferred(model):
    from torch import nn

    seen = set()
    for klass in type(model).__mro__:
        for attr_name, val in vars(klass).items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and hasattr(val, "_deferred_decorator"):
                bound = getattr(model, attr_name)
                setattr(model, attr_name, val._deferred_decorator(bound))

    if isinstance(model, nn.Module):
        for child in model.children():
            apply_deferred(child)
