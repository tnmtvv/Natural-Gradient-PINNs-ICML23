"""Low-rank regularized ENG: batched Jacobian, thin QR and small SVD."""
import argparse
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree


def compressed_svd(J, probes, rank, power_iters):
    # G_E = J.T @ J is applied through J; no p x p Gram matrix is formed.
    Q, _ = jnp.linalg.qr(J.T @ (J @ probes), mode="reduced")
    for _ in range(power_iters):
        Q, _ = jnp.linalg.qr(J.T @ (J @ Q), mode="reduced")
    # JQ = Q_small R; SVD of R gives the compressed curvature spectrum.
    _, R = jnp.linalg.qr(J @ Q, mode="reduced")
    _, singular, Vh = jnp.linalg.svd(R, full_matrices=False)
    return Q @ Vh[:rank].T, singular[:rank]


def inverse_direction(g, V, singular, eps):
    coefficients = V.T @ g
    perpendicular = g - V @ coefficients
    # (eps I + V diag(s^2) V.T)^(-1) g, including the complement.
    return perpendicular / eps + V @ (coefficients / (singular**2 + eps))


def pcg_direction(J, g, V, singular, eps, tolerance, max_iters):
    # Solve with full curvature; low-rank factors only precondition CG.
    floor = eps + singular[-1]**2

    def precondition(vector):
        coefficients = V.T @ vector
        return ((vector - V @ coefficients) / floor
                + V @ (coefficients / (singular**2 + eps)))

    def multiply(vector):
        return J.T @ (J @ vector) + eps*vector

    z = precondition(g)
    threshold = tolerance*jnp.linalg.norm(g)
    initial = (jnp.array(0), jnp.zeros_like(g), g, z, jnp.dot(g, z))

    def condition(state):
        i, x, residual, search, rz = state
        return ((i < max_iters) & (jnp.linalg.norm(residual) > threshold)
                & jnp.isfinite(rz) & (rz > 0))

    def body(state):
        i, x, residual, search, rz = state
        product = multiply(search)
        alpha = rz / jnp.dot(search, product)
        x = x + alpha*search
        residual = residual - alpha*product
        z = precondition(residual)
        new_rz = jnp.dot(residual, z)
        search = z + (new_rz / rz)*search
        return i+1, x, residual, search, new_rz

    iterations, x, _, _, _ = jax.lax.while_loop(condition, body, initial)
    relative_residual = jnp.linalg.norm(g-multiply(x))/jnp.maximum(
        jnp.linalg.norm(g), jnp.finfo(g.dtype).tiny)
    return x, iterations, relative_residual


def run(seed, eps, rank, n_iters, log_every=10, oversampling=8,
        power_iters=1, ls_steps=31, min_step=2.0**-30,
        solver="pcg", cg_tol=1e-10, cg_max_iters=1000, block_size=32,
        compression="psi"):
    from ngrad.models import init_params
    from reproduce_poisson_fig1 import build

    # build() supplies fixed interior/boundary points and weighted residuals.
    ctx = build()
    if "weighted_residuals" not in ctx:
        raise KeyError("Replace build() with the supplied poisson_build.py version")
    params = init_params(ctx["layer_sizes"], random.PRNGKey(seed))
    theta, unravel = ravel_pytree(params)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    residuals = lambda flat: ctx["weighted_residuals"](unravel(flat))
    loss = jax.jit(lambda flat: ctx["loss"](unravel(flat)))
    loss_grad = jax.jit(jax.value_and_grad(loss))
    if "interior_residual" in ctx:
        # Differentiate scalar point residuals, then batch the gradients.
        interior_grad = jax.vmap(jax.grad(
            lambda flat, x: ctx["interior_residual"](unravel(flat), x)
        ), in_axes=(None, 0))
        boundary_grad = jax.vmap(jax.grad(
            lambda flat, x: ctx["model"](unravel(flat), x)
        ), in_axes=(None, 0))

        @jax.jit
        def jacobian(flat):
            return jnp.concatenate([
                jnp.sqrt(2*ctx["interior_weight"])
                * interior_grad(flat, ctx["interior_points"]),
                jnp.sqrt(2*ctx["boundary_weight"])
                * boundary_grad(flat, ctx["boundary_points"]),
            ], axis=0)
    else:
        jacobian = jax.jit(jax.jacrev(residuals))
    error = jax.jit(lambda flat: ctx["rel_l2"](unravel(flat)))
    initial = residuals(theta)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("weighted_residuals must be a nonempty 1D vector")
    if not np.allclose(float(loss(theta)), float(0.5*jnp.vdot(initial, initial)),
                       rtol=1e-8, atol=1e-12):
        raise ValueError("loss must equal 0.5 * ||weighted_residuals||^2")
    rank = min(rank, theta.size, initial.size)
    width = min(rank + oversampling, theta.size, initial.size)
    key = random.PRNGKey(seed + 1)
    steps = jnp.asarray(np.concatenate(([0.0], np.geomspace(1, min_step, ls_steps))))

    @jax.jit
    def factorize(J, probes):
        if solver == "adagram":
            from adagram_block_factor import block_inverse_factor
            return block_inverse_factor(J, rank, eps, block_size, compression)
        return compressed_svd(J, probes, rank, power_iters)

    @jax.jit
    def direction(J, g, V, singular):
        if solver == "adagram":
            from adagram_block_factor import factor_direction
            return factor_direction(g, V, singular, eps), jnp.array(0), jnp.array(jnp.nan)
        if solver == "lowrank":
            return inverse_direction(g, V, singular, eps), jnp.array(0), jnp.array(jnp.nan)
        return pcg_direction(J, g, V, singular, eps, cg_tol, cg_max_iters)

    @jax.jit
    def search(flat, d, current_loss):
        # Sequential device-side grid search avoids Python calls per candidate.
        def body(i, state):
            best_loss, eta = state
            candidate = loss(flat - steps[i]*d)
            better = jnp.isfinite(candidate) & (candidate < best_loss)
            return (jnp.where(better, candidate, best_loss),
                    jnp.where(better, steps[i], eta))
        return jax.lax.fori_loop(1, steps.size, body,
                                (current_loss, jnp.array(0.0)))

    history, timings = [], []
    V_previous = None
    last_eta, fallback = np.nan, False
    for iteration in range(n_iters + 1):
        if iteration % log_every == 0 or iteration == n_iters:
            err = float(error(theta))
            history.append((iteration, err, float(loss(theta)), last_eta, fallback))
            print(f"Low-rank ENG seed {seed} rank {rank} eps {eps:g} "
                  f"iter {iteration:4d} rel L2 {err:.3e}", flush=True)
        if iteration == n_iters:
            break
        started = perf_counter()
        current_loss, g = loss_grad(theta)
        J = jacobian(theta)  # J_R: all weighted residual derivatives at once.
        J.block_until_ready()
        after_jacobian = perf_counter()
        if not bool(jnp.isfinite(current_loss) & jnp.all(jnp.isfinite(g))
                    & jnp.all(jnp.isfinite(J))):
            raise FloatingPointError("Nonfinite loss, gradient or Jacobian")
        if float(jnp.linalg.norm(g)) == 0:
            if history[-1][0] != iteration:
                history.append((iteration, float(error(theta)), float(current_loss), 0, False))
            break
        if solver == "adagram":
            probes = jnp.empty((theta.size, 0), dtype=theta.dtype)
        else:
            key, sample_key = random.split(key)
            probes = random.normal(sample_key, (theta.size, width), dtype=theta.dtype)
            # Reuse the previous subspace as a probe, but recompute current curvature.
            if V_previous is not None:
                probes = probes.at[:, :rank].set(V_previous)
        V, singular = factorize(J, probes)
        d, cg_iterations, cg_residual = direction(J, g, V, singular)
        fallback = ~jnp.all(jnp.isfinite(d)) | ~(jnp.dot(g, d) > 0)
        d = jnp.where(fallback, g, d)
        d.block_until_ready()
        after_factor = perf_counter()
        _, eta = search(theta, d, current_loss)
        last_eta, fallback = float(eta), bool(fallback)
        finished = perf_counter()
        timings.append((iteration, after_jacobian-started,
                        after_factor-after_jacobian, finished-after_factor,
                        int(cg_iterations), float(cg_residual)))
        if iteration < 3 or iteration % log_every == 0:
            print(f"  timings: jacobian/grad {after_jacobian-started:.3f}s, "
                  f"factor/direction {after_factor-after_jacobian:.3f}s, "
                  f"line search {finished-after_factor:.3f}s"
                  + (" (includes JIT compilation)" if iteration == 0 else ""), flush=True)
            if solver == "pcg":
                print(f"  CG iterations {int(cg_iterations)}, relative residual "
                      f"{float(cg_residual):.3e}", flush=True)
        if last_eta == 0:
            print("No decreasing grid step; stopping.", flush=True)
            if history[-1][0] == iteration:
                history[-1] = (iteration, history[-1][1], float(current_loss), 0, fallback)
            else:
                history.append((iteration, float(error(theta)), float(current_loss), 0, fallback))
            break
        theta = theta - eta*d
        V_previous = V
    return np.asarray(history, dtype=float), np.asarray(timings, dtype=float).reshape(-1, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--max-rank", type=int, required=True)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--power-iters", type=int, default=1)
    parser.add_argument("--solver", choices=("pcg", "lowrank", "adagram"), default="pcg")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--compression", choices=("psi", "svd"), default="psi")
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-max-iters", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ls-steps", type=int, default=31)
    parser.add_argument("--min-step", type=float, default=2.0**-30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (not np.isfinite(args.eps) or args.eps <= 0 or args.max_rank < 1
            or args.oversampling < 0 or args.power_iters < 0 or args.iters < 0
            or args.log_every < 1 or args.ls_steps < 2 or not 0 < args.min_step < 1
            or not 0 < args.cg_tol < 1 or args.cg_max_iters < 1 or args.block_size < 1):
        parser.error("Invalid positive eps/rank or iteration/grid parameters")
    history, timings = run(args.seed, args.eps, args.max_rank, args.iters,
                          args.log_every, args.oversampling, args.power_iters,
                          args.ls_steps, args.min_step, args.solver,
                          args.cg_tol, args.cg_max_iters, args.block_size, args.compression)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, iters=history[:, 0].astype(int), rel=history[:, 1],
             loss=history[:, 2], eta=history[:, 3], fallback=history[:, 4].astype(bool),
             timing_iters=timings[:, 0].astype(int), jacobian_seconds=timings[:, 1],
             factor_seconds=timings[:, 2], line_search_seconds=timings[:, 3],
             cg_iterations=timings[:, 4].astype(int), cg_relative_residual=timings[:, 5],
             seed=args.seed, eps=args.eps, rank=args.max_rank,
             oversampling=args.oversampling, power_iters=args.power_iters,
             method={"pcg": "lowrank_preconditioned_eng", "lowrank": "qr_small_svd_eng",
                     "adagram": "adagram_block_inverse_factor_eng"}[args.solver],
             block_size=args.block_size,
             compression=args.compression if args.solver == "adagram" else "none",
             solver=args.solver, cg_tol=args.cg_tol, cg_max_iters=args.cg_max_iters)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
