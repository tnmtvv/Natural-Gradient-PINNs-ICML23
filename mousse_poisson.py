"""Mousse (Muon + Shampoo whitening) on the 2D Poisson PINN.

Single-device version of the update in Anti-Entrophic/Mousse: momentum, Kronecker
statistics, whitening, Newton-Schulz, unwhitening. Biases use AdamW.
"""
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import jax.numpy as jnp
from jax import grad, random
from jax.flatten_util import ravel_pytree

from ngrad.models import init_params
from reproduce_poisson_fig1 import build

NS_CONSTS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def newton_schulz(G, epsilon=1e-7):
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in NS_CONSTS:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.mT
    return X


def clean_eigenvalues(evals, epsilon):
    shift = torch.clamp(-evals.min(), min=0.0) + epsilon
    return evals + shift


class Mousse:
    def __init__(self, params, lr=5e-4, mu=0.95, weight_decay=0.0, epsilon=1e-8, shampoo_epsilon=1e-5, shampoo_beta=0.95, shampoo_update_freq=10, shampoo_alpha=0.125, betas=(0.9, 0.95)):
        self.weights = params["weights"]
        self.biases = params["biases"]
        self.lr = lr
        self.mu = mu
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.shampoo_epsilon = shampoo_epsilon
        self.shampoo_beta = shampoo_beta
        self.shampoo_update_freq = shampoo_update_freq
        self.shampoo_alpha = shampoo_alpha
        self.beta1, self.beta2 = betas
        self.step_count = 0
        self.w_mom = [torch.zeros_like(w) for w in self.weights]
        self.b_mom = [torch.zeros_like(b) for b in self.biases]
        self.b_var = [torch.zeros_like(b) for b in self.biases]
        self.w_state = []
        for w in self.weights:
            m, n = w.shape
            self.w_state.append({
                "L": torch.zeros(m, m, dtype=torch.float32),
                "R": torch.zeros(n, n, dtype=torch.float32),
                "eig_L": (torch.ones(m), torch.eye(m)),
                "eig_R": (torch.ones(n), torch.eye(n)),
            })

    def step(self):
        self.step_count += 1
        beta = self.shampoo_beta
        for w, g, mom, state in zip(self.weights, [p.grad for p in self.weights], self.w_mom, self.w_state):
            mom.mul_(self.mu).add_(g)
            G = g.float()
            M = mom.float()
            m_dim, n_dim = state["L"].shape[0], state["R"].shape[0]
            state["L"].mul_(beta).add_(G @ G.T, alpha=1 - beta)
            state["R"].mul_(beta).add_(G.T @ G, alpha=1 - beta)
            if self.step_count % self.shampoo_update_freq == 1 or self.shampoo_update_freq == 1:
                corr = 1.0 - beta ** self.step_count
                L = state["L"] / corr
                R = state["R"] / corr
                L = L * (m_dim / L.trace().clamp(min=1e-12))
                R = R * (n_dim / R.trace().clamp(min=1e-12))
                eval_L, evec_L = torch.linalg.eigh(L + self.shampoo_epsilon * torch.eye(m_dim))
                eval_R, evec_R = torch.linalg.eigh(R + self.shampoo_epsilon * torch.eye(n_dim))
                state["eig_L"] = (clean_eigenvalues(eval_L, self.shampoo_epsilon), evec_L)
                state["eig_R"] = (clean_eigenvalues(eval_R, self.shampoo_epsilon), evec_R)
            eval_L, evec_L = state["eig_L"]
            eval_R, evec_R = state["eig_R"]
            M = evec_L.T @ M @ evec_R
            scale_L = eval_L.abs().pow(self.shampoo_alpha)
            scale_R = eval_R.abs().pow(self.shampoo_alpha)
            M = M / scale_L.unsqueeze(1) / scale_R.unsqueeze(0)
            M = newton_schulz(M, self.epsilon)
            ns_norm = M.norm()
            M = M / scale_L.unsqueeze(1) / scale_R.unsqueeze(0)
            M = evec_L @ M @ evec_R.T
            M = M * (ns_norm / M.norm().clamp(min=1e-12))
            fan_out, fan_in = w.shape
            adjusted_lr = self.lr * math.sqrt(fan_out / fan_in)
            w.mul_(1 - self.lr * self.weight_decay)
            w.add_(M.to(w.dtype), alpha=-adjusted_lr)

        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count
        for b, mom, var in zip(self.biases, self.b_mom, self.b_var):
            g = b.grad
            mom.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            var.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            b.mul_(1 - self.lr * self.weight_decay)
            b.addcdiv_(mom / bc1, (var / bc2).sqrt().add_(self.epsilon), value=-self.lr)


def run(seed, lr, n_iters, log_every=10):
    ctx = build()
    params0 = init_params(ctx["layer_sizes"], random.PRNGKey(seed))
    weights, biases = [], []
    for w, b in params0:
        weights.append(torch.tensor(np.asarray(w), dtype=torch.float64))
        biases.append(torch.tensor(np.asarray(b), dtype=torch.float64))
    opt = Mousse({"weights": weights, "biases": biases}, lr=lr)
    loss_grad = grad(ctx["loss"])
    rel_l2 = ctx["rel_l2"]
    history = []
    for iteration in range(n_iters + 1):
        jax_params = [(jnp.asarray(w.detach().numpy()), jnp.asarray(b.detach().numpy())) for w, b in zip(weights, biases)]
        if iteration % log_every == 0 or iteration == n_iters:
            err = float(rel_l2(jax_params))
            history.append((iteration, err))
            print(f"Mousse seed {seed} lr {lr:g} iter {iteration:4d} rel L2 {err:.3e}", flush=True)
        if iteration == n_iters:
            break
        gtree = loss_grad(jax_params)
        for (w, b), (gw, gb) in zip(zip(weights, biases), gtree):
            w.grad = torch.tensor(np.asarray(gw), dtype=torch.float64)
            b.grad = torch.tensor(np.asarray(gb), dtype=torch.float64)
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
