import numpy as np
import stan
from math import prod


def constrained_dim(mdl: stan.model.Model) -> int:
    # Note: scalars have empty dimension, but prod([]) is 1
    return sum(map(prod, mdl.dims))


def unconstrained_dim(mdl: stan.model.Model) -> int:
    cdim = constrained_dim(mdl)
    dummy_x = np.zeros(cdim)
    dummy_r = unflatten_constrained_params(mdl, dummy_x)
    return sum([1 if np.isscalar(r) else len(r) for r in dummy_r.values()])


def unflatten_constrained_params(mdl: stan.model.Model, x: np.ndarray) -> dict:
    k = 0
    out = {}
    for para, dim in zip(mdl.param_names, mdl.dims):
        if len(dim) == 0:
            # Scalar case
            out[para] = x[k]
            k += 1
        else:
            # Multi-dimensional case
            out[para] = np.zeros(dim)
            for j in range(out[para].size):
                # item_name for parameter 'x' are things like x.1.2 for 1st row 2nd col
                parts = mdl.constrained_param_names[k + j].split('.')
                assert parts[0] == para, f"Mismatch! {para} != {mdl.constrained_param_names[k + j]}"
                subs = [int(idx)-1 for idx in parts[1:]]
                out[para][subs] = x[k + j]
            k += out[para].size
            out[para] = out[para].tolist()
    assert k == len(mdl.constrained_param_names), "Something went wrong in terms of dimensions"
    return out


def unconstrain_sample(mdl: stan.model.Model, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
        batch_size = tuple()
    else:
        batch_size = x.shape[:1]
    out_size = batch_size + (unconstrained_dim(mdl),)

    if len(mdl.constrained_param_names) != x.shape[1]:
        raise ValueError(f'Only {x.shape[1]} values in x but require {len(mdl.constrained_param_names)} constrained params!')

    return np.stack([
        mdl.unconstrain_pars(unflatten_constrained_params(mdl, s))
        for s in x
    ], axis=0)