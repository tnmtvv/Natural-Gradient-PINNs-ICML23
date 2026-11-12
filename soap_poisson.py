"""SOAP (Shampoo + Adam) on the 2D Poisson PINN."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
from pytorch_optimizer import SOAP

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import jax.numpy as jnp
from jax import grad, random

from ngrad.models import init_params
from reproduce_poisson_fig1 import build


def run(seed, lr, n_iters, log_every=10, weight_decay=0.0):
    ctx = build()
    params0 = init_params(ctx["layer_sizes"], random.PRNGKey(seed))
    tensors = []
    for w, b in params0:
        tensors.append(torch.tensor(np.asarray(w), dtype=torch.float64))
        tensors.append(torch.tensor(np.asarray(b), dtype=torch.float64))
    opt = SOAP(tensors, lr=lr, weight_decay=weight_decay)
    loss_grad = grad(ctx["loss"])
    rel_l2 = ctx["rel_l2"]
    history = []

    def to_jax():
        pairs = []
        for i in range(0, len(tensors), 2):
            pairs.append((jnp.asarray(tensors[i].detach().numpy()), jnp.asarray(tensors[i + 1].detach().numpy())))
        return pairs

    for iteration in range(n_iters + 1):
        jax_params = to_jax()
        if iteration % log_every == 0 or iteration == n_iters:
            err = float(rel_l2(jax_params))
            history.append((iteration, err))
            print(f"SOAP seed {seed} lr {lr:g} iter {iteration:4d} rel L2 {err:.3e}", flush=True)
        if iteration == n_iters:
            break
        grads = loss_grad(jax_params)
        flat_g = []
        for gw, gb in grads:
            flat_g.append(torch.tensor(np.asarray(gw), dtype=torch.float64))
            flat_g.append(torch.tensor(np.asarray(gb), dtype=torch.float64))
        for p, g in zip(tensors, flat_g):
            p.grad = g
        opt.step()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    history = run(args.seed, args.lr, args.iters, args.log_every)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, iters=np.array([i for i, _ in history]), rel=np.array([v for _, v in history]), seed=args.seed, lr=args.lr)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
