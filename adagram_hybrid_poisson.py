"""Hybrid AdaGram schedules from the FCN figure, on the 2D Poisson PINN.

Switch after 50% of steps. The second optimizer keeps P/Q (and Adam momentum
if the warmup optimizer stored it).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import jax.numpy as jnp
from jax import grad, random
from jax.flatten_util import ravel_pytree

from adagram_optimizers.AdaGram_eq import AdaGramEQ
from adagram_optimizers.AdagramPS import AdaGramPS
from adagram_optimizers.AdagramSqrtPS import AdaGramPS_Sqrt
from adagram_optimizers.AdamGram import EQAdamGram, AdamGram, AdamGramSqrt_PS
from ngrad.models import init_params
from reproduce_poisson_fig1 import build

SCHEDULES = {
    "eq": {
        "warmup": EQAdamGram,
        "second": AdaGramEQ,
        "warmup_lr": 0.1,
        "second_lr": 0.1,
        "eps": 1e-2,
    },
    "sqrt": {
        "warmup": AdamGramSqrt_PS,
        "second": AdaGramPS_Sqrt,
        "warmup_lr": 0.1,
        "second_lr": 0.05,
        "eps": 1e-2,
    },
    "adam_ps": {
        "warmup": AdamGram,
        "second": AdaGramPS,
        "warmup_lr": 1.0,
        "second_lr": 1.0,
        "eps": 1.0,
    },
    "adam_sqrt": {
        "warmup": AdamGramSqrt_PS,
        "second": AdaGramPS_Sqrt,
        "warmup_lr": 1.0,
        "second_lr": 1.0,
        "eps": 1.0,
    },
}


def _handoff(old, new, theta):
    src = old.state.get(theta)
    if not src:
        return
    dst = new.state[theta]
    for key in ("P", "Q", "m_t", "step_count", "L_0", "L_0_inv"):
        if key in src:
            dst[key] = src[key]


def run(seed, schedule, n_iters, eps=None, warmup_lr=None, second_lr=None, warmup_eps=None, second_eps=None, log_every=10):
    spec = dict(SCHEDULES[schedule])
    if eps is not None:
        spec["eps"] = eps
        spec["warmup_eps"] = eps
        spec["second_eps"] = eps
    else:
        spec.setdefault("warmup_eps", spec["eps"])
        spec.setdefault("second_eps", spec["eps"])
    if warmup_lr is not None:
        spec["warmup_lr"] = warmup_lr
    if second_lr is not None:
        spec["second_lr"] = second_lr
    if warmup_eps is not None:
        spec["warmup_eps"] = warmup_eps
    if second_eps is not None:
        spec["second_eps"] = second_eps
    ctx = build()
    params = init_params(ctx["layer_sizes"], random.PRNGKey(seed))
    flat0, unravel = ravel_pytree(params)
    theta = torch.tensor(np.asarray(flat0), dtype=torch.float64, requires_grad=False)
    eps = spec["eps"]
    opt = spec["warmup"]([theta], lr=spec["warmup_lr"], eps=spec["warmup_eps"], enable_logging=False)
    loss_grad = grad(ctx["loss"])
    rel_l2 = ctx["rel_l2"]
    switch = n_iters // 2
    history = []
    tag = f"{schedule} seed {seed}"

    for iteration in range(n_iters + 1):
        params = unravel(jnp.asarray(theta.detach().numpy()))
        if iteration % log_every == 0 or iteration == n_iters:
            err = float(rel_l2(params))
            history.append((iteration, err))
            print(f"{tag} iter {iteration:4d} rel L2 {err:.3e}", flush=True)
        if iteration == n_iters:
            break
        if iteration == switch:
            nxt = spec["second"]([theta], lr=spec["second_lr"], eps=spec["second_eps"], enable_logging=False)
            _handoff(opt, nxt, theta)
            opt = nxt
            print(f"{tag} switch at {iteration} lr {spec['second_lr']}", flush=True)
        gflat = ravel_pytree(loss_grad(params))[0]
        theta.grad = torch.tensor(np.asarray(gflat), dtype=torch.float64)
        opt.step()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule", choices=sorted(SCHEDULES), required=True)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--warmup-lr", type=float, default=None)
    parser.add_argument("--second-lr", type=float, default=None)
    parser.add_argument("--warmup-eps", type=float, default=None)
    parser.add_argument("--second-eps", type=float, default=None)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    history = run(
        args.seed,
        args.schedule,
        args.iters,
        eps=args.eps,
        warmup_lr=args.warmup_lr,
        second_lr=args.second_lr,
        warmup_eps=args.warmup_eps,
        second_eps=args.second_eps,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        iters=np.array([i for i, _ in history]),
        rel=np.array([v for _, v in history]),
        seed=args.seed,
        schedule=args.schedule,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
