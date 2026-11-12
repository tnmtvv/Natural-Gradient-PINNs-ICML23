"""Paper's block inverse-factor recurrence with cutoff-selected rank.

F = eps**(-1/2) (I - P Q.T), F.T F approximates (J.T J + eps I)**(-1).
After each exact block append, QR of the two thin factors and SVD of their
small core select the new rank. There is no prescribed rank or rank cap.
This truncates the correction P Q.T, not the Jacobian or inverse operator.

NumPy/SciPy handle changing column counts on the CPU. Unlike a padded JAX
scan, the returned factors really have shape (p, selected_rank). The caller
can still use JAX for the Jacobian, loss and line search.
"""
import math

import numpy as np
from scipy.linalg import solve_triangular


def compress_correction(P, Q, cutoff_rtol):
    """Truncate P Q.T through a core of size at most min(p, r + block).

    Keep sigma_i > 0 and sigma_i >= cutoff_rtol * sigma_max. This is a
    relative singular-value cutoff of the correction; it is not the
    old Gram cutoff p * machine_eps * sigma_max(J)**2. None keeps every
    core mode, including zero modes, for prescribed-width initialization.
    """
    if P.ndim != 2 or P.shape != Q.shape:
        raise ValueError("P and Q must have identical two-dimensional shapes")
    if cutoff_rtol is not None and (not math.isfinite(cutoff_rtol) or not 0 <= cutoff_rtol < 1):
        raise ValueError("cutoff_rtol must be finite and in [0, 1)")
    if P.shape[1] == 0:
        return P.copy(), Q.copy(), dict(rank=0, core_size=0, threshold=0.,
                                       sigma_max=0., discarded_spectral_norm=0.)
    Qp, Rp = np.linalg.qr(P, mode="reduced")
    Qq, Rq = np.linalg.qr(Q, mode="reduced")
    U, singular, Vh = np.linalg.svd(Rp @ Rq.T, full_matrices=False)
    threshold = 0. if cutoff_rtol is None else cutoff_rtol * singular[0]
    keep = (np.ones(singular.shape, dtype=bool) if cutoff_rtol is None else
            (singular > 0.) & (singular >= threshold))
    rank = int(np.count_nonzero(keep))
    updated_P = (Qp @ U[:, :rank]) * singular[:rank]
    updated_Q = Qq @ Vh[:rank].T
    stats = dict(rank=rank, core_size=len(singular), threshold=float(threshold),
                 sigma_max=float(singular[0]),
                 discarded_spectral_norm=float(singular[rank]) if rank < len(singular) else 0.)
    return updated_P, updated_Q, stats


def append_block(P, Q, rows, eps, cutoff_rtol):
    """Equations (5)-(6), followed by cutoff compression of the correction."""
    observed = rows.T
    Z = (observed - P @ (Q.T @ observed)) / np.sqrt(eps)
    B, R = np.linalg.qr(Z, mode="reduced")
    eye = np.eye(R.shape[0], dtype=rows.dtype)
    M = np.linalg.cholesky(eye + R @ R.T)
    left = B @ solve_triangular(M, M - eye, lower=True)
    right = B - Q @ (P.T @ B)
    return compress_correction(np.concatenate((P, left), axis=1),
                               np.concatenate((Q, right), axis=1), cutoff_rtol)


def adaptive_block_inverse_factor(J, eps=1e-12, block_size=32, cutoff_rtol=None,
                                  initial_rank=0):
    """Rebuild factors from current J; return P, Q and per-block statistics.

    Defaults to sqrt(p * machine_epsilon) on correction singular values,
    i.e. sigma(C)**2 >= p * machine_epsilon * sigma_max(C)**2. This uses
    the old Jacobian filter's relative squared-value convention on C.
    The rank can both grow and shrink; only
    the ambient dimension p limits it. If it grows to p, core SVD is p x p.
    No full-Jacobian SVD or dense Gramian is used.

    initial_rank > 0 initializes the correction from that many Jacobian
    rows without cutoff (or all rows if fewer are available). Subsequent
    blocks use the cutoff and can grow beyond this initial rank. Zero
    initial modes keep their columns until the first adaptive compression;
    thus initial_rank is the stored width, not a claim of numerical rank.
    The default zero preserves the original empty-factor initialization.
    """
    J = np.asarray(J)
    if J.ndim != 2 or J.shape[1] < 1 or not np.issubdtype(J.dtype, np.floating):
        raise ValueError("J must be a floating-point matrix with positive width")
    if not np.all(np.isfinite(J)):
        raise ValueError("J must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    width = J.shape[1]
    if not isinstance(initial_rank, (int, np.integer)) or not 0 <= initial_rank <= width:
        raise ValueError("initial_rank must be an integer between zero and the parameter count")
    if cutoff_rtol is None:
        cutoff_rtol = np.sqrt(width * np.finfo(J.dtype).eps)
    if not math.isfinite(cutoff_rtol) or not 0 <= cutoff_rtol < 1:
        raise ValueError("cutoff_rtol must be finite and in [0, 1)")
    P = np.empty((width, 0), dtype=J.dtype)
    Q = np.empty_like(P)
    history = []
    initial_rows = min(initial_rank, len(J))
    if initial_rows:
        P, Q, stats = append_block(P, Q, J[:initial_rows], eps, None)
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
            raise FloatingPointError("Nonfinite inverse factors during initialization")
        history.append(dict(rows_consumed=initial_rows, previous_rank=0,
                            cutoff_rtol=0., initialization=True, **stats))
    for start in range(initial_rows, len(J), block_size):
        rows = J[start:start + block_size]
        previous_rank = P.shape[1]
        P, Q, stats = append_block(P, Q, rows, eps, cutoff_rtol)
        if not np.all(np.isfinite(P)) or not np.all(np.isfinite(Q)):
            raise FloatingPointError(f"Nonfinite inverse factors after row {start}")
        history.append(dict(rows_consumed=start + len(rows), previous_rank=previous_rank,
                            cutoff_rtol=float(cutoff_rtol), initialization=False, **stats))
    return P, Q, history


def adaptive_factor_direction(g, P, Q, eps=1e-12):
    """Same F.T @ F @ g application as factor_direction; no extra filter."""
    y = (g - P @ (Q.T @ g)) / np.sqrt(eps)
    return (y - Q @ (P.T @ y)) / np.sqrt(eps)
