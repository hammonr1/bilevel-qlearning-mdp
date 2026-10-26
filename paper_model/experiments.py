"""Cached, parallel experiment runner. Each job = one learning run (Algorithm 1 for one allocation,
one parameter set, one seed). Per-seed summaries go to results/raw/<name>.json (committed),
Q-tables and visited-state sets to results/qtables/<name>.pkl.gz (kept on disk, git-ignored)."""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .env import Environment
from .instance import generate_instance
from .learner import train_and_evaluate
from .params import Params

ROOT = Path(__file__).resolve().parent / "results"
RAW = ROOT / "raw"
QT = ROOT / "qtables"


def params_to_dict(p: Params) -> dict:
    return dataclasses.asdict(p)


def params_from_dict(d: dict) -> Params:
    fixed = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    return Params(**fixed)


def job_name(exp: str, params: Params, sites, seed: int, n_train: int, n_eval: int) -> str:
    return f"{exp}__{params.label()}__tb-{params.tie_break}__s{'-'.join(map(str, sites))}__seed{seed}__n{n_train}-{n_eval}"


def make_spec(exp: str, params: Params, sites, seed: int, n_train: int, n_eval: int, **extra) -> dict:
    sites = tuple(sorted(int(j) for j in sites))
    return dict(name=job_name(exp, params, sites, seed, n_train, n_eval), exp=exp,
                params=params_to_dict(params), sites=sites, seed=seed, n_train=n_train, n_eval=n_eval, **extra)


def run_spec(spec: dict, save_qtables: bool = True) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    QT.mkdir(parents=True, exist_ok=True)
    raw = RAW / f"{spec['name']}.json"
    if raw.exists():
        with open(raw) as f:
            return json.load(f)
    params = params_from_dict(spec["params"])
    inst = generate_instance(7)
    env = Environment(inst, params, spec["sites"])
    t0 = time.perf_counter()
    res = train_and_evaluate(env, params, spec["n_train"], spec["n_eval"], spec["seed"])
    out = dict(spec)
    out.update(res.summary())
    out.update(sourcing_solver=env.sourcing_solver_name, covered=inst.covered_count(spec["sites"]),
               wall_s=time.perf_counter() - t0, eval_losses=res.eval_losses,
               train_curve=res.train_losses)
    if save_qtables:
        with gzip.open(QT / f"{spec['name']}.pkl.gz", "wb", compresslevel=5) as f:
            pickle.dump(dict(QA=res.QA, QD=res.QD, visited_A=list(res.visited_A), visited_D=list(res.visited_D),
                             spec=spec), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp = raw.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, raw)
    return out


def run_many(specs: list, workers: int = int(os.environ.get("PM_WORKERS", "8")), label: str = "", log=sys.stderr) -> list:
    """Run all specs (cached ones return immediately); results in the same order as specs."""
    pending = [s for s in specs if not (RAW / f"{s['name']}.json").exists()]
    results = {}
    if pending:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_spec, s): s["name"] for s in pending}
            for fut in as_completed(futs):
                r = fut.result()
                results[r["name"]] = r
                done += 1
                if done % 25 == 0 or done == len(pending):
                    el = time.time() - t0
                    print(f"  [{label}] {done}/{len(pending)} runs, {el:.0f}s elapsed, "
                          f"~{el / done * (len(pending) - done):.0f}s left", file=log, flush=True)
    out = []
    for s in specs:
        if s["name"] in results:
            out.append(results[s["name"]])
        else:
            with open(RAW / f"{s['name']}.json") as f:
                out.append(json.load(f))
    return out
