"""Compare SCFD epsilon values at fixed rank/block size and iteration budget.

Reuses the validated eps=1e-10 baseline when settings and source hashes match.
All new runs call the unchanged benchmark_scfd.py from the same initialization.
Two independent runs may execute concurrently; their timings are not a fair
speed comparison with the saved baseline. Accuracy uses the same seed and budget.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pinn-scfd-mpl')

import numpy as np

from tune_scfd import write_json

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'results/scfd_tuning_eps1e-10_seed0/rank64_block16_800'


def read_run(folder, eps, args, initial=None, source_hashes=None):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Source changed since saved run: {name}')
    if source_hashes is not None and manifest['source_sha256'] != source_hashes:
        raise ValueError(f'Source mismatch: {folder}')
    with np.load(folder / 'scfd.npz', allow_pickle=False) as archive:
        data = dict(archive)
    for name, value in [('rank', args.rank), ('block_size', args.block_size), ('seed', args.seed),
                        ('eps', eps), ('final_iteration', args.iters), ('eval_points', 95**2)]:
        if data[name].item() != value:
            raise ValueError(f'{name} does not match in {folder}')
    if initial is not None:
        np.testing.assert_array_equal(data['initial_theta'], initial)
    np.testing.assert_array_equal(data['update'], np.arange(1, args.iters + 1))
    for key in ['relative_l2', 'loss', 'eta', 'final_theta', 'compensation', 'block_history']:
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f'Nonfinite {key} in {folder}')
    np.testing.assert_array_equal(data['effective_ridge'], eps + data['compensation'])
    assert data['final_relative_l2'] == data['relative_l2'][-1]
    assert np.all(data['sketch_rank'] == args.rank)
    blocks = data['block_history']
    assert np.all(blocks[:, 4] <= min(args.rank + args.block_size, int(data['parameter_count'])))
    for i in range(1, args.iters + 1):
        block = blocks[blocks[:, 0] == i]
        assert block[0, 6] == 0 and block[-1, 2] == 1020
        np.testing.assert_allclose(block[:, 6], np.cumsum(block[:, 5]), rtol=1e-15)
        assert block[-1, 6] == data['compensation'][i - 1]
    return data, manifest['source_sha256']


def summarize(folder, data, reused=False):
    return dict(eps=float(data['eps']), relative_l2=float(data['final_relative_l2']),
                loss=float(data['loss'][-1]), final_compensation=float(data['compensation'][-1]),
                final_effective_ridge=float(data['effective_ridge'][-1]),
                training_seconds=float(data['update_seconds'][-1]),
                path=str(folder), reused=reused,
                loss_increases=int(np.sum(np.diff(np.append(data['loss_before'], data['loss'][-1])) > 1e-12)))


def run_candidate(out, eps, args, initial, hashes):
    name = f'eps{eps:.12g}'
    folder = out / name
    command = [sys.executable, str(ROOT / 'benchmark_scfd.py'), '--rank', str(args.rank),
               '--block-size', str(args.block_size), '--iters', str(args.iters),
               '--eps', str(eps), '--seed', str(args.seed), '--out-dir', str(folder)]
    print(f'Start eps={eps:g}, rank={args.rank}, block={args.block_size}, updates={args.iters}', flush=True)
    with (out / f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    data, _ = read_run(folder, eps, args, initial, hashes)
    return summarize(folder, data)


def plot_results(out, runs, args, initial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for row in sorted(runs, key=lambda row: row['eps']):
        with np.load(Path(row['path']) / 'scfd.npz', allow_pickle=False) as data:
            label = f"eps={row['eps']:g}" + (' (saved)' if row['reused'] else '')
            line, = axes[0].semilogy(data['iteration'], data['relative_l2'], label=label, linewidth=1.8)
            axes[1].semilogy(data['update'], data['effective_ridge'],
                             color=line.get_color(), linewidth=1.5, label=label)
    cutoff_path = ROOT / 'results/adaptive_factor_cutoff_initial20_eps1e-10_800/adaptive_factor.npz'
    if cutoff_path.exists():
        with np.load(cutoff_path, allow_pickle=False) as data:
            if int(data['seed']) == args.seed and np.array_equal(data['initial_theta'], initial):
                mask = data['iteration'] <= args.iters
                axes[0].semilogy(data['iteration'][mask], data['relative_l2'][mask], '--', color='black',
                                 label='Saved correction cutoff, eps=1e-10', alpha=.8)
    axes[0].set(xlabel='Iteration', ylabel='Relative L2 error')
    axes[1].set(xlabel='Iteration', ylabel=r'Effective diagonal shift $\varepsilon+\alpha$')
    for ax in axes:
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'SCFD SVD: rank {args.rank}, block {args.block_size}, '
                 f'seed {args.seed}, {args.iters} updates\n'
                 'Only epsilon changes; sketch and compensation reset at every update')
    fig.savefig(out / 'epsilon_comparison.png', dpi=170)
    fig.savefig(out / 'epsilon_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eps-values', type=float, nargs='+', default=[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6])
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=800)
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--out-dir', type=Path, default=Path('results/scfd_eps_sweep_rank64_block16_seed0'))
    args = parser.parse_args()
    eps_values = list(dict.fromkeys(args.eps_values))
    if (not all(math.isfinite(eps) and eps > 0 for eps in eps_values)
            or min(args.rank, args.block_size, args.iters) < 1 or not 0 <= args.seed < 2**32):
        parser.error('Invalid epsilon or experiment settings')
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose an empty output directory')
    baseline_folder = args.baseline.resolve()
    with np.load(baseline_folder / 'scfd.npz', allow_pickle=False) as baseline:
        baseline_eps = float(baseline['eps'])
    data, hashes = read_run(baseline_folder, baseline_eps, args)
    initial = data['initial_theta']
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'search_plan.json', dict(eps_values=eps_values, rank=args.rank,
                block_size=args.block_size, seed=args.seed, iterations=args.iters, jobs=args.jobs,
                baseline=str(baseline_folder), source_sha256=hashes,
                sweep_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection_metric='final relative L2 after the same 800 updates' if args.iters == 800
                                 else 'final relative L2 after equal iteration counts',
                timing_note='Concurrent new runs; timings are not a controlled speed comparison with saved runs.'))
    runs = [summarize(baseline_folder, data, reused=True)] if baseline_eps in eps_values else []
    failures = []
    pending = [eps for eps in eps_values if eps != baseline_eps]
    write_json(out / 'results.json', runs)
    write_json(out / 'status.json', dict(state='running', completed=len(runs), total=len(eps_values)))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_candidate, out, eps, args, initial, hashes): eps for eps in pending}
        for future in as_completed(futures):
            eps = futures[future]
            try:
                result = future.result()
                runs.append(result)
                print(json.dumps(result), flush=True)
            except Exception as error:
                failure = dict(eps=eps, error=repr(error))
                failures.append(failure)
                print(json.dumps(failure), flush=True)
            write_json(out / 'results.json', sorted(runs, key=lambda row: row['eps']))
            write_json(out / 'status.json', dict(state='running', completed=len(runs),
                                                total=len(eps_values), failures=failures))
    runs.sort(key=lambda row: row['eps'])
    winner = min(runs, key=lambda row: row['relative_l2']) if runs else None
    report = dict(runs=runs, failures=failures, best=winner, rank=args.rank, block_size=args.block_size,
                  seed=args.seed, iterations=args.iters,
                  note='Single-seed epsilon comparison; best means lowest final relative L2 among tested values.')
    write_json(out / 'summary.json', report)
    if runs:
        with (out / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
        plot_results(out, runs, args, initial)
    write_json(out / 'status.json', dict(state='completed' if not failures else 'completed_with_failures',
                                        completed=len(runs), total=len(eps_values), failures=failures))


if __name__ == '__main__':
    main()
