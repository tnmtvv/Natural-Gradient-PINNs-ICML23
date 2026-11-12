"""Grid-search AdaGramVanilla (lr, eps) on the 2D Poisson PINN and record curves."""
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
from adagram_optimizers.AdagramVanilla import AdaGramVanilla
from adagram_optimizers.AdamGram import AdamGram, AdamGramSqrt_PS
from ngrad.models import init_params
from reproduce_poisson_fig1 import build

OPTIMIZERS = {
    "vanilla": AdaGramVanilla,
    "eq": AdaGramEQ,
    "adam": AdamGram,
    "ps": AdaGramPS,
    "sqrt": AdaGramPS_Sqrt,
    "adam_sqrt": AdamGramSqrt_PS,
}


def run(seed, lr, eps, n_iters, log_every=10, opt_name="vanilla", max_rank=None):
    ctx = build()
    params = init_params(ctx["layer_sizes"], random.PRNGKey(seed))
    flat0, unravel = ravel_pytree(params)
    theta = torch.tensor(np.asarray(flat0), dtype=torch.float64)
    opt = OPTIMIZERS[opt_name]([theta], lr=lr, eps=eps, max_rank=max_rank, enable_logging=False)
    loss_grad = grad(ctx["loss"])
    rel_l2 = ctx["rel_l2"]

    history = []
    for iteration in range(n_iters + 1):
        params = unravel(jnp.asarray(theta.detach().numpy()))
        if iteration % log_every == 0 or iteration == n_iters:
            err = float(rel_l2(params))
            history.append((iteration, err))
            print(
                f"AdaGram-{opt_name} seed {seed} lr {lr:g} eps {eps:g} iter {iteration:4d} rel L2 {err:.3e}",
                flush=True,
            )
        if iteration == n_iters:
            break
        gflat = ravel_pytree(loss_grad(params))[0]
        theta.grad = torch.tensor(np.asarray(gflat), dtype=torch.float64)
        opt.step()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--opt", choices=sorted(OPTIMIZERS), default="vanilla")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    history = run(args.seed, args.lr, args.eps, args.iters, args.log_every, args.opt, args.max_rank)
    idx = np.array([i for i, _ in history])
    val = np.array([v for _, v in history])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, iters=idx, rel=val, seed=args.seed, lr=args.lr, eps=args.eps)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
