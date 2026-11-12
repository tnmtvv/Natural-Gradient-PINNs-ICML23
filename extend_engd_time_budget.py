"""Continue paired E-NGD runs to the SVD wall-time budget, preserving originals."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_cross_equation import METHODS, fingerprint, make_optimizer, summarize, write_json
from cross_equation_problems import build_problem


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/cross_equation_pilot"))
    args = parser.parse_args()
    base = args.out_dir.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    settings = manifest["settings"]
    if settings["source_sha256"] != fingerprint():
        raise ValueError("Original benchmark sources changed; cannot continue consistently")
    if json.loads((base / "status.json").read_text())["state"] != "completed":
        raise ValueError("Finish the original benchmark before timing continuations")
    out = base / "time_matched"
    out.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(out / "manifest.json", dict(original_run_id=manifest["run_id"],
               continuation_source_sha256=source_hash, original_manifest=str(base / "manifest.json"),
               meaning="E-NGD continued to max(SVD,E-NGD) time. All other runs unchanged; same initialization and batch sequence.",
               timing=manifest["timing"]))
    records = {}
    status = dict(state="running", completed=0, total=len(settings["budgets"])*len(settings["seeds"]))
    try:
        for name, budget in settings["budgets"].items():
            problem = build_problem(name, small=settings["small"])
            _, update = make_optimizer(problem, "engd", settings["eps"])
            for seed in settings["seeds"]:
                status.update(problem=name, seed=seed)
                write_json(out / "status.json", status)
                for method in METHODS:
                    data = read(base / name / f"seed_{seed}" / f"{method}.npz")
                    if data["run_id"].item() != manifest["run_id"]:
                        raise ValueError("Mismatched original run")
                    records[name, seed, method] = data
                old = records[name, seed, "engd"]
                target = max(float(old["train_seconds"]), float(records[name, seed, "svd_cut"]["train_seconds"]))
                folder = out / name / f"seed_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "engd.npz"
                if path.exists():
                    data = read(path)
                    if (data["run_id"].item() != manifest["run_id"]
                            or data["continuation_source_sha256"].item() != source_hash
                            or float(data["time_budget_seconds"]) != target):
                        raise ValueError("Incompatible continuation")
                else:
                    theta = jnp.asarray(old["final_theta"])
                    iteration = int(old["final_iteration"])
                    elapsed = float(old["train_seconds"])
                    started = time.perf_counter()
                    jax.block_until_ready(update(theta, (), problem.batch(iteration)))
                    jax.block_until_ready(problem.metrics(theta))
                    warmup = time.perf_counter()-started
                    rows = old["history"].tolist()
                    while elapsed < target:
                        started = time.perf_counter()
                        batch = problem.batch(iteration)
                        theta, _, info = jax.block_until_ready(update(theta, (), batch))
                        elapsed += time.perf_counter()-started
                        iteration += 1
                        if iteration % settings["log_every"] == 0 or elapsed >= target:
                            l2, h1 = np.asarray(problem.metrics(theta))
                            rows.append([iteration, elapsed, l2, h1, float(problem.loss(theta, batch)), *np.asarray(info)])
                    history = np.asarray(rows)
                    if not np.isfinite(history).all() or not np.isfinite(theta).all():
                        raise FloatingPointError("Non-finite continuation")
                    data = dict(old, history=history, final_theta=np.asarray(theta),
                                final_iteration=iteration, train_seconds=elapsed,
                                final_relative_l2=history[-1, 2], final_relative_h1=history[-1, 3],
                                continuation_warmup_seconds=warmup, time_budget_seconds=target,
                                continuation_source_sha256=source_hash)
                    with path.with_suffix(".tmp").open("wb") as stream:
                        np.savez(stream, **data)
                    path.with_suffix(".tmp").replace(path)
                records[name, seed, "engd"] = data
                print(f"DONE {name} seed={seed} E-NGD equal time: {int(data['final_iteration'])} updates, "
                      f"L2={float(data['final_relative_l2']):.6e}, {float(data['train_seconds']):.2f}s", flush=True)
                status["completed"] += 1
            summarize(out, records, settings["seeds"], settings["budgets"])
            del update, problem
            jax.clear_caches()
        status["state"] = "completed"
        write_json(out / "status.json", status)
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
