"""Batched inverse-factor update: arXiv:2510.19349v2, Theorem 2, Appendix D.2.

The first `rank` Jacobian rows are appended exactly. That factor is
orthonormalized by one SVD, and every later block keeps this rank by
projector splitting or a fresh SVD. F = eps^(-1/2)(I - P Q.T).
"""
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular


def compress_svd(U, S, V, left, right, rank):
    P = jnp.concatenate((U @ S, left), axis=1)
    Q = jnp.concatenate((V, right), axis=1)
    Qp, Rp = jnp.linalg.qr(P, mode="reduced")
    Qq, Rq = jnp.linalg.qr(Q, mode="reduced")
    small_U, singular, Vh = jnp.linalg.svd(Rp @ Rq.T, full_matrices=False)
    return Qp @ small_U[:, :rank], jnp.diag(singular[:rank]), Qq @ Vh[:rank].T


def compress_psi(U, S, V, left, right):
    # Projector splitting for Delta C = left @ right.T.
    delta_V = left @ (right.T @ V)
    U1, S_hat = jnp.linalg.qr(U @ S + delta_V, mode="reduced")
    S_middle = S_hat - U1.T @ delta_V
    V1, S_transpose = jnp.linalg.qr(
        V @ S_middle.T + right @ (left.T @ U1), mode="reduced")
    return U1, S_transpose.T, V1


def _inverse_update(Z):
    # Appendix D.2: Z = B R, H = I + R R.T = M M.T, left = B (I - M^{-1}).
    B, R = jnp.linalg.qr(Z, mode="reduced")
    eye = jnp.eye(R.shape[0], dtype=Z.dtype)
    factor = jnp.linalg.cholesky(eye + R @ R.T)
    gain = solve_triangular(factor, factor - eye, lower=True)
    return B @ gain, B


def _append_exact(U, V, filled, rows, eps):
    # Raw factors, C = U V.T. New columns land in [filled, filled + width).
    active = (jnp.arange(U.shape[1]) < filled).astype(rows.dtype)
    previous_U = U * active
    previous_V = V * active
    observed = rows.T
    left, base = _inverse_update(
        (observed - previous_U @ (previous_V.T @ observed)) / jnp.sqrt(eps))
    right = base - previous_V @ (previous_U.T @ base)
    width = rows.shape[0]
    destination = filled + jnp.arange(width, dtype=jnp.int32)
    slots = jnp.arange(U.shape[1])
    selector = (destination[:, None] == slots[None, :]).astype(rows.dtype)
    return (U + left @ selector, V + right @ selector,
            filled + jnp.int32(width))


def _orthonormalize(U, V):
    # Reparameterize C = U V.T as U_hat S V_hat.T with orthonormal bases.
    left_basis, values, left_coeff = jnp.linalg.svd(U, full_matrices=False)
    core = (values[:, None] * left_coeff) @ V.T
    core_left, singular, core_right = jnp.linalg.svd(core, full_matrices=False)
    return left_basis @ core_left, jnp.diag(singular), core_right.T


def _maintain(state, rows, eps, compression, rank):
    U, S, V = state
    observed = rows.T
    left, base = _inverse_update(
        (observed - U @ (S @ (V.T @ observed))) / jnp.sqrt(eps))
    right = base - V @ (S.T @ (U.T @ base))
    if compression == "svd":
        updated = compress_svd(U, S, V, left, right, rank)
    else:
        updated = compress_psi(U, S, V, left, right)
    return updated


def _consume(rows, block_size, apply, state):
    count = rows.shape[0]
    full, remainder = divmod(count, block_size)
    if full:
        blocks = rows[:full * block_size].reshape(full, block_size, rows.shape[1])
        state, _ = lax.scan(lambda carry, block: (apply(carry, block), None),
                             state, blocks)
    if remainder:
        state = apply(state, rows[full * block_size:])
    return state


def block_inverse_factor(J, rank, eps, block_size, compression="psi"):
    if compression not in ("psi", "svd"):
        raise ValueError("compression must be psi or svd")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    count, width = J.shape
    rank = min(rank, width)
    if rank < 1:
        raise ValueError("rank must be positive")
    eps = jnp.asarray(eps, dtype=J.dtype)
    exact_count = min(count, rank)
    exact = (jnp.zeros((width, rank), dtype=J.dtype),
             jnp.zeros((width, rank), dtype=J.dtype), jnp.int32(0))

    def append(state, rows):
        U, V, filled = state
        return _append_exact(U, V, filled, rows, eps)

    U, V, _ = _consume(J[:exact_count], block_size, append, exact)
    if count < rank:
        return U, V
    basis = _orthonormalize(U, V)

    def maintain(state, rows):
        return _maintain(state, rows, eps, compression, rank)

    U, S, V = _consume(J[rank:], block_size, maintain, basis)
    return U @ S, V


def factor_direction(g, P, Q, eps):
    y = (g - P @ (Q.T @ g)) / jnp.sqrt(eps)
    return (y - Q @ (P.T @ y)) / jnp.sqrt(eps)
