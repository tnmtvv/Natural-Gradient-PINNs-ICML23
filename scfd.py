"""Block Spectral Compensation Frequent Directions for the energy Gramian.

Reference: Chen et al., IJCAI 2020, Algorithm 1 and Section 3.2,
https://www.ijcai.org/proceedings/2020/0588.pdf.

For a retained rank r, shrink squared singular values by sigma_{r+1}**2
after each block append, retain r rows, and accumulate the full shrinkage
in alpha. This is the block FD variant with paper sketch parameter m=r+1;
it compresses at block boundaries rather than using the paper's 2m buffer.
The approximation is B.T B + (eps + alpha) I. SCFD operates on Jacobian
rows, NOT on the generally nonsymmetric inverse correction P Q.T.
Rebuild B and alpha for every current Jacobian; do not carry them across
optimization updates. No numerical spectral cutoff is applied.
"""
import math

import numpy as np


def _row_svd(rows):
    """SVD through a square core of width at most the row count."""
    basis, core = np.linalg.qr(rows.T, mode="reduced")
    _, singular, right = np.linalg.svd(core.T, full_matrices=False)
    return singular, right @ basis.T


def compress_scfd(B, rows, compensation, rank):
    """Append rows and return the shrunk sketch, cumulative alpha and stats.

    In exact arithmetic, with A = [B; rows] and delta = sigma_{r+1}(A)^2,
    0 <= A.T A - B_new.T B_new <= delta I. Adding delta I therefore
    compensates the entire lost spectrum, including all discarded modes.
    """
    B, rows = np.asarray(B), np.asarray(rows)
    if (B.ndim != 2 or rows.ndim != 2 or B.shape[1] != rows.shape[1]
            or B.shape[1] < 1 or B.dtype != rows.dtype
            or not np.issubdtype(B.dtype, np.floating)
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(rows))):
        raise ValueError("B and rows must be finite floating matrices of the same width and dtype")
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= B.shape[1]:
        raise ValueError("rank must be an integer between one and the parameter count")
    if not math.isfinite(compensation) or compensation < 0:
        raise ValueError("compensation must be finite and nonnegative")
    stacked = np.concatenate((B, rows), axis=0)
    if len(stacked) == 0:
        return B.copy(), float(compensation), dict(rank=0, core_size=0, delta=0.,
                                                  compensation=float(compensation), sigma_max=0.)
    singular, right = _row_svd(stacked)
    with np.errstate(over='raise', invalid='raise'):
        squared = singular**2
        delta = float(squared[rank]) if rank < len(squared) else 0.
        retained = np.maximum(squared[:rank] - delta, 0.)
        updated = np.sqrt(retained)[:, None] * right[:rank]
        alpha = float(compensation + delta)
    if not np.isfinite(alpha):
        raise FloatingPointError("SCFD compensation overflow")
    stats = dict(rank=len(updated), core_size=len(singular), delta=delta,
                 compensation=alpha, sigma_max=float(singular[0]))
    return updated, alpha, stats


def scfd_sketch(J, rank=20, block_size=32):
    """Sketch current J, initializing from its first rank rows without loss.

    Returned B has at most rank rows (possibly zero-valued); rank denotes
    stored width, not numerical rank. History includes initialization and
    each subsequent row block. The dense Gramian is never constructed.
    """
    J = np.asarray(J)
    if (J.ndim != 2 or J.shape[1] < 1 or not np.issubdtype(J.dtype, np.floating)
            or not np.all(np.isfinite(J))):
        raise ValueError("J must be a finite floating-point matrix with positive width")
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= J.shape[1]:
        raise ValueError("rank must be an integer between one and the parameter count")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    B = np.empty((0, J.shape[1]), dtype=J.dtype)
    alpha, history = 0., []
    initial_count = min(rank, len(J))
    stops = [(0, initial_count, True)] if initial_count else []
    stops += [(start, min(start + block_size, len(J)), False)
              for start in range(initial_count, len(J), block_size)]
    for start, stop, initialization in stops:
        B, alpha, stats = compress_scfd(B, J[start:stop], alpha, rank)
        history.append(dict(rows_consumed=stop, initialization=initialization, **stats))
    return B, alpha, history


def scfd_direction(g, B, compensation, eps=1e-10):
    """Apply (B.T B + (eps + alpha) I)^(-1) to the loss gradient.

    Resolve the sketch span and its orthogonal complement separately to
    avoid subtracting nearly equal inverse terms in the Woodbury formula.
    A second projection removes roundoff leakage back into the sketch span.
    """
    B, g = np.asarray(B), np.asarray(g)
    if (B.ndim != 2 or g.shape != (B.shape[1],)
            or not np.issubdtype(B.dtype, np.floating)
            or not np.issubdtype(g.dtype, np.floating)
            or not np.all(np.isfinite(B)) or not np.all(np.isfinite(g))):
        raise ValueError("B and g must be finite floating arrays with matching dimensions")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(compensation) or compensation < 0:
        raise ValueError("compensation must be finite and nonnegative")
    ridge = eps + compensation
    if not math.isfinite(ridge):
        raise ValueError("eps + compensation must be finite")
    if not len(B):
        return g / ridge
    singular, right = _row_svd(B)
    coefficients = right @ g
    if len(right) == B.shape[1]:
        perpendicular = np.zeros_like(g)
    else:
        perpendicular = g - right.T @ coefficients
        leakage = right @ perpendicular
        perpendicular -= right.T @ leakage
        coefficients += leakage
    return right.T @ (coefficients / (singular**2 + ridge)) + perpendicular / ridge
