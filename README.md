**Achieving High Accuracy with PINNs via Energy Natural Gradient Descent ICML 2023**

Topic: *Github Repo for the ICML 2023 paper: Energy Natural gradient methods for PINNs and Deep Ritz.*

Paper: [arxiv](https://arxiv.org/abs/2302.13163)

Abstract: *We propose energy natural gradient descent, a natural gradient method with respect to a Hessian-induced Riemannian metric as an optimization algorithm for physics-informed neural networks (PINNs) and the deep Ritz method. As a main motivation we show that the update direction in function space resulting from the energy natural gradient corresponds to the Newton direction modulo an orthogonal projection onto the model's tangent space. We demonstrate experimentally that energy natural gradient descent yields highly accurate solutions with errors several orders of magnitude smaller than what is obtained when training PINNs with standard optimizers like gradient descent, Adam or BFGS, even when those are allowed significantly more computation time. We show that the approach can be combined with deterministic and stochastic discretizations of the integral terms and with deep networks allowing for an application in higher dimensional settings.*

## Requirements
- python 3.10.10 or later
- jax 0.4.8 or later
- jaxopt 0.6 or later
- optax 0.1.4 or later

## Installation
All required packages (jax, jaxopt, optax) are pip installable. For GPU support the jax version must be compatible with the CUDA version and the NVIDIA driver installed on the machine.

## Usage
The four main examples of the paper (Poisson 2d, Poisson 5d, Heat 1d, Nonlinear 1d) can be found in the top level scripts with the corresponding names. If the required packages are installed these scripts can directly be executed. The prefix of the scripts indicates which solver is used, with engd corresponding to the energy natural gradient descent. For brevity, the Adam, gradient descent (gd) and BFGS optimizers are only included for the 2d Poisson example. 

## PSI inverse-factor Poisson variant

Run `python engd_psi_poisson_2d.py --metric energy --rank 32 --eps 1e-6 --block-size 32`.
Use `--metric l2` for untransformed parameter derivatives of the model in
both interior and boundary quadrature. This changes the energy metric.

The energy variant preserves the original model, quadrature, loss, errors,
and line-search grid (2**(-k), k=0,...,30). Its weighted Jacobian is built
from parameter derivatives of u_theta, with the spatial Laplacian applied
in the interior. J.T @ J equals the original discretized Gramian.
The objective gradient remains grad(loss).

The existing adagram_block_factor.py implements the batched inverse-factor
updates and PSI compression from https://arxiv.org/abs/2510.19349.
The direction is F.T @ F @ grad(loss), with
F = eps**(-0.5) * (I - P @ Q.T), approximating (G + eps*I)**(-1).
Positive regularization differs from the original least-squares pseudoinverse.
Factors are rebuilt from the current Jacobian at each optimization step;
accumulating across steps would instead invert a historical sum of Gramians.
The rectangular Jacobian is still stored; only the inverse factor is compressed.

To add the full-rank SVD comparison to the saved 800-iteration benchmark:

```bash
python benchmark_psi_poisson.py --compression svd --full-rank --block-size 32 --metric energy --seed 0 --iters 800 --out-dir results/psi_800 --no-plot
python plot_psi_poisson.py --out-dir results/psi_800 --metric energy --x-axis iterations
```

`--full-rank` uses the parameter count (257 for this model), retaining all
singular components during the block updates, with the same regularization
`eps=1e-6`. The plot includes all saved SVD curves in this directory automatically.

## Adaptive rank of the inverse-factor correction

`adagram_adaptive_factor.py` keeps the same block inverse-factor recurrence
from the paper, but replaces a prescribed rank with a cutoff after every
block. It starts with empty `P, Q`, appends the block correction, takes QR
of the two factors, and decomposes only `Rp @ Rq.T`. Singular values of
`C = P @ Q.T` below `cutoff_rtol * sigma_max(C)` are dropped. The returned
factor column count really changes; it can shrink and grow. No rank cap is
imposed beyond the parameter dimension. If the selected rank approaches
that dimension, the core decomposition can also become parameter-sized.

```bash
python benchmark_adaptive_factor_cutoff.py --eps 1e-12 --iters 100 --out-dir results/adaptive_factor_cutoff_pilot
```

To start each Jacobian pass at rank 20, initialize from its first 20 rows
without cutoff, then let the remaining 32-row updates grow or shrink the rank:

```bash
python benchmark_adaptive_factor_cutoff.py --initial-rank 20 --eps 1e-12 --iters 100 --out-dir results/adaptive_factor_cutoff_initial20
```

`--initial-rank` is an initialization, not a rank cap. Its default is zero
(empty initialization). `initial_pass_rank.png` shows the actual column count
after each block of the first optimization update, including initialization.

The default relative cutoff is `sqrt(p * machine_epsilon)` on the singular
values of the correction (about `2.39e-7` for p=257, float64).
`--cutoff-rtol` overrides it. Epsilon is ridge regularization and is independent
of this relative cutoff. The old Jacobian cutoff uses
`sigma(J)**2 >= p * machine_epsilon * sigma_max(J)**2`; the new default applies
that same *relative squared-singular-value convention* to C. A weaker numerical
cutoff `--cutoff-rtol 5.706546346573305e-14` can also be tested for p=257.
These choices act on C, not on J or G, and need not reproduce the old blue
curve. Removing a correction mode leaves the baseline inverse `1/eps` in
that direction, whereas the old spectral pseudoinverse cutoff gives it zero
weight.

Changing array shapes are handled on the CPU with NumPy/SciPy; JAX still
computes the current Jacobian, loss, and the original line search. Factors
are rebuilt for each current Jacobian. `adaptive_factor.npz`, `block_ranks.csv`,
`manifest.json`, and `summary.json` record the trajectory, actual rank at
every block, source hashes, thresholds, and timing. `comparison.png` overlays
compatible saved reference curves; it does not recompute a full Jacobian SVD.
Use an empty output directory. Defaults are seed 0, 800 updates, block size
32 and `eps=1e-12`; the 100-update command above is a pilot.

Run the numerical checks with `python -m unittest test_adagram_adaptive_factor -v`.
They verify the block inverse against a direct regularized solve, invariance
under factor rescaling, genuine rank growth/shrinkage, and small-core SVD use.

## Spectral Compensation Frequent Directions (SCFD)

```bash
python benchmark_scfd.py --rank 20 --eps 1e-10 --block-size 32 --seed 0 --iters 800 --out-dir results/scfd_rank20_eps1e-10_800
python -m unittest test_scfd -v
```

`scfd.py` implements the block form of SCFD from
[Chen et al., IJCAI 2020, Algorithm 1](https://www.ijcai.org/proceedings/2020/0588.pdf).
It streams the weighted Jacobian rows into a sketch `B`. After appending
each block, SVD through a QR core gives singular values `s`; it sets
`delta = s[rank]**2` (the first discarded value), keeps
`sqrt(max(s[:rank]**2 - delta, 0))`, and adds `delta` to `alpha`.
If there is no discarded value, `delta=0`. The first `rank` rows initialize
the sketch without shrinkage. No relative or absolute spectral cutoff is used.
Here `rank` is the **maximum retained width**, unlike the cutoff experiment's
`initial_rank`; the paper's shrinkage index is `m=rank+1`. This variant
compresses at each block boundary, rather than using the paper's `2m` buffer.

The direction solves `(B.T @ B + (eps + alpha)*I) d = grad(loss)`.
SCFD sketches the positive semidefinite Gramian through Jacobian rows;
its compensation is not applied to the nonsymmetric inverse correction `P Q.T`.
Both `B` and `alpha` are reset for each optimization update. Epsilon remains
`1e-10`, while the effective diagonal shift is `eps+alpha` and can be much
larger. The model, quadrature, gradient and line search match the cutoff run.
The full Jacobian is still materialized, but no dense Gramian or full-Jacobian
SVD is computed. Core sizes are at most `min(p, rank+block_size)`.

Results include `scfd.npz`, per-block `blocks.csv`, source hashes and settings
in `manifest.json`, and final metrics in `summary.json`. `comparison.png`
and `.pdf` overlay compatible saved cutoff, full-rank SVD and E-NGD runs;
`compensation.png` shows the effective diagonal shift and step sizes.
Reference curves are saved runs, not new executions. Use an empty output folder.
Tests check known spectral shrinkage, covariance bounds and monotonicity,
the direction against a dense solve, full-rank exactness, and core size bounds.

For a bounded hyperparameter search, run:

```bash
python tune_scfd.py --out-dir results/scfd_tuning_eps1e-10_seed0
```

This compares retained ranks 32, 64, 96 and 128 at block size 32 for
100 updates, then tests block sizes 16 and 64 at the best pilot rank.
It chooses the lowest final relative L2 error at that shared pilot budget
and reruns that configuration from the original initialization for 800
updates. Epsilon stays at `1e-10`. Override `--ranks`, `--blocks`,
`--pilot-iters`, `--full-iters`, `--eps` or `--seed` for other searches.
Each candidate uses the unchanged `benchmark_scfd.py`, writes its own
trajectory and source hashes, and has a separate log. `pilots.csv`,
`selection.json` and `tuning_comparison.png` summarize the comparison.
This is a coordinate search on one seed, selected using the known exact
solution; it does not establish robustness across initializations or find
a global optimum. SCFD compensation remains the full accumulated shrinkage.

To compare epsilon values at the selected rank 64 and block size 16:

```bash
python sweep_scfd_eps.py --eps-values 1e-12 1e-11 1e-10 1e-9 1e-8 1e-6 --iters 800
```

All candidates use the same seed and initialization for the same iteration
budget. The matching saved `eps=1e-10` run is reused only after its settings
and source hashes are checked. `--baseline` selects that saved run folder;
it must match `--rank`, `--block-size`, `--seed` and `--iters`. New runs use
the unchanged SCFD benchmark. By default two run concurrently (`--jobs 1`
selects sequential execution), so timings should not be used as a controlled
speed comparison against the saved run. `comparison.csv` and
`epsilon_comparison.png`/`.pdf` compare final errors, trajectories, and
effective regularization `eps+alpha`. Each run is validated for identical
initial parameters, matching source code, finite outputs, and cumulative
compensation. The sweep requires an empty output directory and preserves
all earlier results.

## Switched ENGD / full-rank SVD benchmark

```bash
python benchmark_hybrid_poisson.py --iters 800 --switch-after 50 --seed 0 --eps 1e-6 --block-size 32 --out-dir results/hybrid_800
```

Runs both ENGD → SVD and SVD → ENGD from the same initialization, using
the existing energy-metric benchmark and its line search. Updates 1–50
use the first method; updates 51–800 use the second method, continuing
from the current network parameters. SVD retains all 257 components of
the block inverse factor and uses `eps=1e-6`; ENGD uses the unregularized
least-squares solve. Both update functions are compiled before timing.

The output directory contains per-update errors, losses, step sizes,
active methods and cumulative training times in two NPZ files, including
initial/final parameter vectors and run metadata. `summary.json` records
switch/final errors; `hybrid_comparison.png` and `.pdf` compare convergence
by update count and synchronized training time (excluding compilation and
metric evaluation).

To run only ENGD → SVD with an earlier switch, preserving the 50-update results:

```bash
python benchmark_hybrid_poisson.py --order engd-svd --switch-after 25 --iters 800 --out-dir results/hybrid_switch25_800
```

## Spectral cutoff comparison across five seeds

```bash
python benchmark_spectral_cutoff_seeds.py --seeds 0 1 2 3 4 --iters 800 --out-dir results/spectral_cutoff_5seeds
```

Runs the existing function-diagnostics solvers from identical initial parameters
within each seed: SVD with spectral cutoff at `eps=1e-10` and `eps=1e-12`,
full-rank block SVD at `eps=1e-10`, and the E-NGD reference. The model has
257 parameters; quadrature and PDE-loss line search are unchanged.

Each seed gets raw NPZ results and a comparison plot. After all runs finish,
`relative_l2_median_iqr.png`/`.pdf` show the median with a 25th–75th percentile
band, and `relative_l2_all_seeds.png`/`.pdf` show individual seeds. The output
also includes `aggregate.csv`, final errors in `summary.json`, source hashes
and environment details in `manifest.json`, and progress in `status.json`.
The final error is evaluated after all 800 updates. Rerunning the same command
resumes completed runs after checking their settings and source hashes;
`--plot-only` rebuilds plots from completed results.

To plot pointwise nominal 95% bootstrap intervals for the same five-seed
medians without rerunning training:

```bash
python plot_spectral_cutoff_ci.py --out-dir results/spectral_cutoff_5seeds
```

This writes `relative_l2_median_ci95.png`/`.pdf`, `aggregate_ci95.csv`, and
`summary_ci95.json`. It enumerates all 3125 bootstrap resamples of complete
seed trajectories and uses percentile bounds. With only five seeds, these
bounds equal the observed min/max; under iid continuous sampling their
population-median coverage is 93.75%, so 95% is nominal, not exact. These
are pointwise intervals, not a simultaneous band over all iterations.

## Transfer to Heat 1D, nonlinear Deep Ritz, and Poisson 5D

```bash
python verify_cross_equation.py
python benchmark_cross_equation.py --seeds 0 1 2 --out-dir results/cross_equation_pilot
```

The new benchmark reuses `ridge_cut_direction` from the Poisson diagnostics:
an SVD of the weighted energy Jacobian, cutoff
`p * machine_epsilon * sigma_max**2`, and ridge denominator
`sigma**2 + 1e-12` on retained modes. It is not the block inverse-factor
variant. For the nonlinear problem the factor has rows corresponding to
`d_x d_theta u` and `sqrt(3) * u * d_theta u`, so its Gramian is the Hessian
of the **function-space energy**, not a squared PDE residual objective.

Local baselines are E-NGD (`lstsq(J.T @ J, grad(loss))`), JAXopt BFGS,
Optax Adam with the original learning-rate schedule, and GD with the original
31-point line-search grid. Network widths, initializations, quadrature weights,
and loss normalization follow the repository's three original example scripts.
Analytic spatial derivatives of the shallow tanh network are checked against
those scripts using JAX autodiff; parameter derivatives still use autodiff.

The default pilot uses three initialization seeds, 2000 natural-gradient updates
for Heat, 500 for nonlinear Deep Ritz, and 201 for Poisson 5D. Each conventional
optimizer runs for at least the same number of updates and until it reaches the
larger measured SVD/E-NGD training time for that seed (cap: 200000 updates).
Timing synchronizes JAX and excludes compilation, warmup, and error evaluation.
The first plot column compares equal update counts; the second shows training
time. Shading denotes an empirical IQR, not a confidence interval. Final metrics
include the last completed update, as well as the common update-budget endpoint.

Poisson 5D passes fresh batches explicitly into the compiled functions. All
methods and seeds receive the same sequence of random batches; evaluation uses
an independent fixed set of 40000 points. This avoids capturing mutable
integrator points in a JIT closure. Stochastic BFGS refreshes the current loss
and gradient on each new batch before its line search, retaining its inverse
Hessian estimate. These details differ from blindly executing the old scripts.

`results/cross_equation_pilot` contains raw NPZ trajectories and initial/final
parameters, `final_results.csv`, `summary.json`, `comparison.png`/`.pdf`,
`manifest.json` with settings, versions and source hashes, and live progress in
`status.json`. Completed runs resume only if settings and source hashes match.
Use a new output directory after changing the implementation or configuration.

Published median L2 errors from Tables 3 and 5 of
[the paper](https://arxiv.org/html/2302.13163v2) are saved separately in
`paper_reference.json`, including the H-NGD reference. Its Hilbert metric is not
specified by the corresponding repository scripts, so this benchmark does not
silently substitute an arbitrary Hilbert baseline. Figure 3 supplies no numeric
final-error table for Poisson 5D, so no values are inferred from its pixels.
The completed pilot also includes `comparison_to_paper.csv`, with separate
columns for common-update local errors, matched-time local errors, and published
medians. Blank published cells indicate that no numeric table value is available.

This pilot is **not an exact reproduction of the paper**: it uses three rather
than ten initializations and a local CPU, keeps the repository's quadratures
(e.g. 4000 rather than the paper's 3000 interior points in 5D), and gives Adam/GD
matched training time rather than always 200000 updates. `--paper-budget`
requests 200000 Adam/GD updates; it does not remove the other differences.
The reported H1 error follows the repository's sum-of-seminorms convention.

After the pilot completes, give E-NGD the full SVD time budget as well:

```bash
python extend_engd_time_budget.py --out-dir results/cross_equation_pilot
```

This continues each saved E-NGD trajectory from its final parameters and next
batch index, preserving all original results. `time_matched/` holds the nine
continuations and a new comparison plot, CSV, and summary. Its time panel and
final-error table compare all five methods at matched training times (up to
one update's overshoot, unless a conventional method reaches the update cap).
The iteration panel remains restricted to the original common update budget.
Warmup and evaluation are still excluded. Continuations are timed sequentially
after the original experiments, so CPU load variation is a limitation of this
pilot's timing results.

To show every saved trajectory through its actual final time, without rerunning
training:

```bash
python plot_cross_equation_full_time.py --out-dir results/cross_equation_pilot
```

`time_matched/comparison_full_time.png`/`.pdf` set each problem's time axis to
its longest measured run. Thin lines show all seed trajectories and dots mark
their final measurements. Thick medians and IQR bands stop when the first seed
ends, keeping the same seed cohort throughout; individual curves continue to
their actual endpoints without extrapolation. `comparison_full_time_all_seeds`
provides separate panels for each seed. Exact axis limits and per-run durations
are recorded in `comparison_full_time_metadata.json`.

Add `--seed 0` to show only seed 0, with full measured curves and no averaging
or percentile bands. This writes `comparison_full_time_seed_0.png`/`.pdf` and
uses that seed's longest run for each problem's time-axis limit. Seeds 1 and 2
can be selected in the same way.

For iteration-only plots of the same single-seed results, run
`python plot_cross_equation_iterations.py --seed 0`. This saves
`time_matched/comparison_iterations_seed_0.png`/`.pdf` on the original common
update budgets, and `comparison_full_iterations_seed_0.png`/`.pdf` with all
saved updates (logarithmic x-axis above iteration 1). Both use raw trajectories
without averaging across seeds; E-NGD's continuation appears in the full view.

Use `python plot_cross_equation_iterations.py --aggregate` to reproduce the
original three-seed median/IQR iteration panels in one horizontal row, keeping
the original 2000/500/201 update budgets and the original interpolation grid.
Outputs: `time_matched/comparison_iterations_median_iqr_horizontal.png`/`.pdf`.

For a fast execution check only, use tiny grids in a separate directory:

```bash
python benchmark_cross_equation.py --small --iters 2 --seeds 0 --log-every 1 --max-baseline-iters 5 --out-dir results/cross_equation_smoke
```
