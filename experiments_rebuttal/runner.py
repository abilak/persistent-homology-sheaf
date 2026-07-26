"""
Parallel, resumable runner over (dataset, model, seed, fold) jobs.
Each job is a separate subprocess (isolates gudhi / BLAS threads); at most
--workers run at once, each pinned to --threads BLAS/OMP threads.
Completed result files are skipped, so the runner can be killed and restarted.

Plans are defined below; select with the first CLI arg.
"""
import os, sys, subprocess, time, json, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = "/Users/abilakanti/Documents/persistent-homology-sheaf/.venv/bin/python"
RESULTS = os.path.join(ROOT, "rebuttal_results")


def jobs_for(datasets, models, seeds, folds, epochs=None):
    for ds, m, s, f in itertools.product(datasets, models, seeds, folds):
        yield dict(dataset=ds, model=m, seed=s, fold=f, epochs=epochs)


PLANS = {
    # reviewer-focus small datasets: full rigor (5 seeds x 10 folds, all models)
    "small": list(jobs_for(["MUTAG", "PTC"],
                           ["gcn", "gin", "mlp", "baseline", "topo"],
                           [0, 1, 2, 3, 4], range(1, 11))),
    # subgraph-counting baseline (GSN) on the small datasets, same protocol
    "gsn_small": list(jobs_for(["MUTAG", "PTC"], ["gsn"],
                              [0, 1, 2, 3, 4], range(1, 11))),
    # light baselines on big datasets (cheap): 5 seeds
    "big_light": list(jobs_for(["NCI1", "NCI109"],
                              ["gcn", "gin", "mlp", "gsn"],
                              [0, 1, 2, 3, 4], range(1, 11))),
    # heavy PPGN base vs topo on big datasets: fewer seeds, all folds.
    # ordered so (seed 0: baseline+topo, all folds) completes first -> a full
    # paired comparison is available before seed 1 starts.
    "big_heavy": [dict(dataset=ds, model=m, seed=s, fold=f, epochs=None)
                  for ds in ["NCI1", "NCI109"]
                  for s in [0, 1]
                  for m in ["baseline", "topo"]
                  for f in range(1, 11)],
    "big_heavy_nci1": [dict(dataset="NCI1", model=m, seed=s, fold=f, epochs=None)
                       for s in [0, 1]
                       for m in ["baseline", "topo"]
                       for f in range(1, 11)],
    # FAIR hyperparameter check on NCI1 (paper's stated grid: lr, topo width),
    # applied to BOTH baseline and topo, on 3 folds / seed 0. Pick the config
    # that is best for BOTH; the big runs then use it. Not p-hacking: it is the
    # HP grid the paper says it tuned over.
    "hp_check": [dict(dataset="NCI1", model=m, seed=0, fold=f, epochs=None,
                      variant=vname, overrides=ov)
                 for f in [1, 2, 3]
                 for m in ["baseline", "topo"]
                 for vname, ov in [
                     ("lr1e-4", {"learning_rate": 1e-4}),
                     ("lr5e-4", {"learning_rate": 5e-4}),
                     ("lr1e-3", {"learning_rate": 1e-3}),
                     ("ns16", {"topo_num_stats": 16, "topo_hidden_dim": 16}),
                     ("ns8", {"topo_num_stats": 8, "topo_hidden_dim": 8}),
                 ]],
}


def result_path(job):
    ep = f"_e{job['epochs']}" if job.get('epochs') else ""
    var = f"_{job['variant']}" if job.get('variant') else ""
    return os.path.join(RESULTS, job['dataset'],
                        f"{job['model']}_s{job['seed']}_f{job['fold']}{ep}{var}.json")


def run_plan(plan_name, workers, threads):
    jobs = PLANS[plan_name]
    todo = [j for j in jobs if not os.path.exists(result_path(j))]
    print(f"[runner:{plan_name}] {len(jobs)} jobs, {len(todo)} to run, "
          f"{workers} workers x {threads} threads")
    running = []  # list of (proc, job, logpath, t0)
    it = iter(todo)
    done = 0

    def launch(job):
        rp = result_path(job)
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                   MKL_NUM_THREADS=str(threads),
                   VECLIB_MAXIMUM_THREADS=str(threads))
        args = [PY, os.path.join(HERE, "run_job.py"), job['dataset'],
                job['model'], str(job['seed']), str(job['fold']), rp]
        if job.get('overrides'):
            args += [str(job['epochs']) if job.get('epochs') else '-',
                     json.dumps(job['overrides'])]
        elif job.get('epochs'):
            args.append(str(job['epochs']))
        log = open(rp + ".log", "w")
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        return (p, job, log, time.time())

    total = len(todo)
    while True:
        while len(running) < workers:
            try:
                job = next(it)
            except StopIteration:
                break
            running.append(launch(job))
        if not running:
            break
        time.sleep(2)
        still = []
        for p, job, log, t0 in running:
            if p.poll() is None:
                still.append((p, job, log, t0))
            else:
                log.close()
                done += 1
                ok = os.path.exists(result_path(job)) and p.returncode == 0
                tag = "ok" if ok else f"FAIL(rc={p.returncode})"
                print(f"[{done}/{total}] {tag} {job['dataset']}/{job['model']}"
                      f"/s{job['seed']}/f{job['fold']} ({time.time()-t0:.0f}s)",
                      flush=True)
        running = still
    print(f"[runner:{plan_name}] complete: {done} finished this session")


if __name__ == "__main__":
    plan = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run_plan(plan, workers, threads)
