"""Bounded SCFD rank/block search, followed by an 800-update confirmation.

Uses the existing benchmark unchanged. Selection uses relative L2 error at
the same pilot iteration on one seed; it is not a multi-seed robustness test.
Each candidate starts from the same original network initialization.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pinn-scfd-mpl')
ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_candidate(out, rank, block, iters, eps, seed):
    folder = out / f'rank{rank}_block{block}_{iters}'
    command = [sys.executable, str(ROOT / 'benchmark_scfd.py'), '--rank', str(rank),
               '--block-size', str(block), '--iters', str(iters), '--eps', str(eps),
               '--seed', str(seed), '--out-dir', str(folder)]
    print(f'Start rank={rank}, block={block}, iterations={iters}', flush=True)
    write_json(out / 'status.json', dict(state='running', rank=rank, block_size=block,
                                        iterations=iters, candidate=str(folder)))
    with (out / f'{folder.name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((folder / 'summary.json').read_text())
    result = dict(rank=rank, block_size=block, iterations=iters,
                  relative_l2=summary['final_relative_l2'], loss=summary['final_loss'],
                  compensation=summary['final_compensation'],
                  training_seconds=summary['training_seconds'], path=str(folder.relative_to(ROOT)))
    print(json.dumps(result), flush=True)
    return result


def plot_search(out, candidates, winner, pilot_iters, seed, eps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for candidate in candidates:
        with np.load(ROOT / candidate['path'] / 'scfd.npz', allow_pickle=False) as data:
            axes[0].semilogy(data['iteration'], data['relative_l2'],
                             label=f"r={candidate['rank']}, block={candidate['block_size']}")
    baseline_path = ROOT / 'results/scfd_rank20_eps1e-10_800/scfd.npz'
    if baseline_path.exists():
        with np.load(baseline_path, allow_pickle=False) as baseline:
            if int(baseline['seed']) == seed and float(baseline['eps']) == eps:
                mask = baseline['iteration'] <= pilot_iters
                axes[0].semilogy(baseline['iteration'][mask], baseline['relative_l2'][mask],
                                 ':', color='black', label='Previous r=20, block=32')
                axes[1].semilogy(baseline['iteration'], baseline['relative_l2'],
                                 ':', color='black', label='Previous SCFD r=20, block=32')
    with np.load(ROOT / winner['path'] / 'scfd.npz', allow_pickle=False) as data:
        axes[1].semilogy(data['iteration'], data['relative_l2'], linewidth=2,
                         label=f"Selected SCFD r={winner['rank']}, block={winner['block_size']}")
    cutoff = ROOT / 'results/adaptive_factor_cutoff_initial20_eps1e-10_800/adaptive_factor.npz'
    if cutoff.exists():
        with np.load(cutoff, allow_pickle=False) as data:
            if int(data['seed']) == seed:
                axes[1].semilogy(data['iteration'], data['relative_l2'], '--', color='tab:red',
                                 label='Saved correction cutoff, eps=1e-10')
    for ax in axes:
        ax.set(xlabel='Iteration', ylabel='Relative L2 error')
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_title(f'Equal-budget pilot: {pilot_iters} updates')
    axes[1].set_title(f"Selected configuration: {winner['iterations']} updates")
    fig.suptitle(f'SCFD rank/block search, seed {seed}, eps={eps:g}')
    fig.savefig(out / 'tuning_comparison.png', dpi=170)
    fig.savefig(out / 'tuning_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, nargs='+', default=[32, 64, 96, 128])
    parser.add_argument('--blocks', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--pilot-iters', type=int, default=100)
    parser.add_argument('--full-iters', type=int, default=800)
    parser.add_argument('--eps', type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=Path('results/scfd_tuning_eps1e-10_seed0'))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if (not args.ranks or min(args.ranks + args.blocks + [args.pilot_iters, args.full_iters]) < 1
            or args.full_iters <= args.pilot_iters):
        parser.error('Positive settings and full-iters > pilot-iters are required')
    if out.exists() and any(out.iterdir()):
        parser.error('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'search_plan.json', dict(rank_candidates=args.ranks, rank_search_block_size=32,
               block_candidates=args.blocks, pilot_iterations=args.pilot_iters,
               full_iterations=args.full_iters, eps=args.eps, seed=args.seed,
               selection_metric='final relative L2 at equal pilot iteration count',
               scope='one-seed coordinate search, all runs restart from identical initial parameters'))
    candidates = []
    try:
        for rank in dict.fromkeys(args.ranks):
            candidates.append(run_candidate(out, rank, 32, args.pilot_iters, args.eps, args.seed))
            write_json(out / 'pilots.json', candidates)
        best_rank = min(candidates, key=lambda row: row['relative_l2'])['rank']
        for block in dict.fromkeys(args.blocks):
            if block == 32:
                continue
            candidates.append(run_candidate(out, best_rank, block, args.pilot_iters, args.eps, args.seed))
            write_json(out / 'pilots.json', candidates)
        selected = min(candidates, key=lambda row: row['relative_l2'])
        write_json(out / 'selection.json', selected)
        winner = run_candidate(out, selected['rank'], selected['block_size'], args.full_iters,
                               args.eps, args.seed)
        report = dict(pilots=candidates, selected_pilot=selected, full_run=winner,
                      note='Selection and confirmation use seed 0 by default; multiple seeds are needed for robustness.')
        write_json(out / 'summary.json', report)
        with (out / 'pilots.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
        plot_search(out, candidates, winner, args.pilot_iters, args.seed, args.eps)
        write_json(out / 'status.json', dict(state='completed', **winner))
    except BaseException as error:
        write_json(out / 'status.json', dict(state='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
