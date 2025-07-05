# this file imports utilities for functorch for compatibility
try:
    # noinspection PyUnresolvedReferences
    from torch.func import jacfwd, jacrev, vmap, hessian
except ModuleNotFoundError:
    from functorch import jacfwd, jacrev, vmap, hessian

# noinspection PyUnresolvedReferences
from functorch.compile import (
    memory_efficient_fusion,
    aot_module,
    ts_compile,
    min_cut_rematerialization_partition,
)
