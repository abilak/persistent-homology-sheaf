"""
Run EVERY remaining experiment of the paper (all but ogbg-molhiv) from ONE
queue, then the timing benchmark:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
        python experiments_rebuttal/run_everything.py 3 < /dev/null

One scheduler instead of a chain of plans: workers never sit idle at the end
of a plan, and per-group caps keep memory-heavy jobs from running together
(PROTEINS: 1 at a time, since a batch of its 620-node graphs peaks near 9 GB;
BREC: 1 at a time). Order: cheap and critical first (interpretability,
expressivity reruns, BREC started early because it is long), then the main
tables, then the appendix studies, then timing alone on the GPU.

Every job is skipped if its output exists, so the script can be stopped and
restarted at any time; failed jobs are retried once at the end of the queue.
Logs: <result>.log next to each result, and rebuttal_results/run_everything.log.
"""
import os, sys, json, time, shutil, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)
import runner

PY = os.environ.get("PY", sys.executable)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
WORKERS = int(ARGS[0]) if ARGS else 3
DRY = "--dry" in sys.argv
# --skip=name1,name2: drop plans / scripts whose name starts with any of these
#   e.g. --skip=brec_,srg_classification,ablation_min_fast,zinc_ablation_fast
SKIP = tuple(s for a in sys.argv if a.startswith("--skip=") for s in a[7:].split(",") if s)
# --timing-first: run the timing benchmark (alone on the GPU) before the queue
TIMING_FIRST = "--timing-first" in sys.argv
CAPS = {"proteins": 1, "brec": 1}
EXPR = os.path.join("rebuttal_results", "expr_final")
BREC_OUT = os.path.join("rebuttal_results", "brec_final")
BREC_DATA = os.path.join("experiments_rebuttal", "data", "brec_v3.npy")
LOG = open(os.path.join("rebuttal_results", "run_everything.log"), "a")
os.makedirs(EXPR, exist_ok=True); os.makedirs(BREC_OUT, exist_ok=True)


def say(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ---------------------------------------------------------------- job list
MAIN_PLANS = [  # main tables (no molhiv)
    "core_fast", "extra_tu_fast", "zinc_fast_s10",
    "wide_fast", "wide_extra_tu_fast", "wide_zinc_fast_s10", "wide_fast_proteins",
    "mp_fast", "mp_zinc_fast_s10",
    "static_fast", "static_zinc_fast_s10", "static_fast_proteins", "static_extra_tu_fast"]
STUDY_PLANS = [  # appendix studies
    "structure_fast", "structure_fast_proteins", "placement_fast", "placement_zinc_fast_s10",
    "ablation_min_fast", "zinc_ablation_fast",
    "depth_mutag", "depth_zinc", "depth_ptc", "depth_nci1"]


def plan_jobs(names):
    jobs, seen = [], set()
    for name in names:
        for j in runner.PLANS[name]:
            rp = runner.result_path(j)
            if rp in seen:
                continue
            seen.add(rp)
            args = [PY, os.path.join(HERE, "run_job.py"), j["dataset"], j["model"],
                    str(j["seed"]), str(j["fold"]), rp]
            if j.get("overrides"):
                args += [str(j["epochs"]) if j.get("epochs") else "-", json.dumps(j["overrides"])]
            elif j.get("epochs"):
                args.append(str(j["epochs"]))
            jobs.append(dict(name=f"{name}:{j['dataset']}/{j['model']}/s{j['seed']}/f{j['fold']}",
                             cmd=args, done=rp, log=rp + ".log",
                             group="proteins" if j["dataset"] == "PROTEINS" else "light"))
    return jobs


def script_job(name, cmd, done, group="light", copy=None):
    return dict(name=name, cmd=[PY] + cmd, done=done, log=os.path.join(EXPR, name + ".log"),
                group=group, copy=copy)


def expressivity_jobs():
    jobs = [
        script_job("interpret_mutag", ["experiments_rebuttal/interpret_mutag.py"],
                   "rebuttal_results/interpret_mutag/stats.json"),
        # two more training seeds: is the learned chemical ordering reproducible?
        *[script_job(f"interpret_mutag_seed{s}", ["experiments_rebuttal/interpret_mutag.py", f"--seed={s}"],
                     f"rebuttal_results/interpret_mutag_seed{s}/stats.json") for s in (1, 2)],
        # strongly regular pairs, final model at D = 3, P = 2
        script_job("srg_analysis", ["experiments_rebuttal/srg_analysis.py"],
                   os.path.join(EXPR, "srg_analysis.done"), copy="experiments_rebuttal/srg_results.json"),
        script_job("srg_seeds", ["experiments_rebuttal/srg_seeds.py"],
                   os.path.join(EXPR, "srg_seeds.done"), copy="experiments_rebuttal/srg_seeds_results.json"),
        script_job("count_vs_ph", ["experiments_rebuttal/count_vs_ph.py"],
                   os.path.join(EXPR, "count_vs_ph.done"), copy="experiments_rebuttal/count_vs_ph_results.json"),
        script_job("srg_classification", ["experiments_rebuttal/srg_classification.py", "--seeds=3"],
                   os.path.join(EXPR, "srg_classification.done"), copy="rebuttal_results/srg_classification.json"),
    ]
    if os.path.exists(BREC_DATA):
        for m in ("topo", "ppgn", "gin"):        # longest first
            out = os.path.join(BREC_OUT, f"{m}.json")
            jobs.append(script_job(f"brec_{m}", ["experiments_rebuttal/brec_expressivity.py", "--model", m,
                                                 "--data", BREC_DATA, "--sample-num", "400",
                                                 "--epochs", "20", "--out", out], out, group="brec"))
    return jobs


def fetch_brec():
    """BREC's 400 pairs (GraphPKU/BREC, BREC_data_all.zip) -> experiments_rebuttal/data/brec_v3.npy"""
    if os.path.exists(BREC_DATA):
        return True
    tmp = os.path.join("rebuttal_results", "_brec_repo")
    try:
        if not os.path.isdir(tmp):
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/GraphPKU/BREC.git", tmp],
                           check=True, capture_output=True, timeout=600)
        z = glob.glob(os.path.join(tmp, "**", "BREC_data_all.zip"), recursive=True)
        if z:
            subprocess.run(["unzip", "-o", "-q", z[0], "-d", os.path.join(tmp, "data")], check=True, timeout=600)
        found = glob.glob(os.path.join(tmp, "**", "brec_v3.npy"), recursive=True)
        if not found:
            raise FileNotFoundError("brec_v3.npy not found in the BREC repository")
        os.makedirs(os.path.dirname(BREC_DATA), exist_ok=True)
        shutil.copy(found[0], BREC_DATA)
        say(f"BREC data -> {BREC_DATA}")
        return True
    except Exception as e:
        say(f"WARNING: could not fetch BREC data ({type(e).__name__}: {e}); BREC runs skipped. "
            f"Put brec_v3.npy at {BREC_DATA} and restart to include them.")
        return False


# ---------------------------------------------------------------- scheduler
def run_queue(jobs, workers, caps):
    todo = [j for j in jobs if not os.path.exists(j["done"])]
    say(f"{len(jobs)} jobs, {len(todo)} to run, {workers} workers, caps {caps}")
    running, done_n, retried, t_start = [], 0, set(), time.time()
    while todo or running:
        # launch as many as allowed, in priority order, respecting group caps
        launched = True
        while launched and len(running) < workers:
            launched = False
            active = {}
            for r in running:
                active[r["job"]["group"]] = active.get(r["job"]["group"], 0) + 1
            for k, j in enumerate(todo):
                if active.get(j["group"], 0) < caps.get(j["group"], workers):
                    todo.pop(k)
                    os.makedirs(os.path.dirname(j["log"]) or ".", exist_ok=True)
                    lf = open(j["log"], "w")
                    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    p = subprocess.Popen(j["cmd"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
                    running.append(dict(p=p, job=j, lf=lf, t0=time.time()))
                    launched = True
                    break
        time.sleep(5)
        still = []
        for r in running:
            if r["p"].poll() is None:
                still.append(r); continue
            r["lf"].close(); j = r["job"]; done_n += 1
            ok = r["p"].returncode == 0
            if ok and j.get("copy") and os.path.exists(j["copy"]):
                shutil.copy(j["copy"], os.path.join(EXPR, os.path.basename(j["copy"])))
            if ok and j["done"].endswith(".done"):
                open(j["done"], "w").write("ok\n")
            ok = ok and os.path.exists(j["done"])
            el = time.time() - r["t0"]
            if not ok and j["name"] not in retried:
                retried.add(j["name"]); todo.append(j)       # one retry, at the end of the queue
                tag = "FAIL (will retry)"
            else:
                tag = "ok" if ok else "FAIL"
            left = len(todo) + len(still)
            say(f"[{done_n}] {tag} {j['name']} ({el/60:.1f} min)  | {left} left, "
                f"{(time.time()-t_start)/3600:.1f} h elapsed  | log: {j['log']}")
        running = still


def main():
    say("=== run_everything start")
    if not DRY:
        fetch_brec()
    # the long expressivity runs (SR classification, BREC) go last: the paper tables come first,
    # and on a machine short of memory the long runs should not hold slots for hours
    expr = expressivity_jobs()
    LONG = ("srg_classification", "brec_")
    quick = [j for j in expr if not j["name"].startswith(LONG)]
    long_ = [j for j in expr if j["name"].startswith(LONG)]
    jobs = quick + plan_jobs(MAIN_PLANS) + plan_jobs(STUDY_PLANS) + long_
    if SKIP:
        jobs = [j for j in jobs if not j["name"].split(":")[0].startswith(SKIP)]
        say(f"skipping {SKIP}")
    if DRY:
        todo = [j for j in jobs if not os.path.exists(j["done"])]
        by = {}
        for j in todo:
            key = j["name"].split(":")[0] if ":" in j["name"] else j["name"]
            by[key] = by.get(key, 0) + 1
        say(f"DRY RUN: {len(jobs)} jobs, {len(todo)} to run; per plan/script: {by}")
        say(f"groups: proteins {sum(j['group'] == 'proteins' for j in todo)}, brec {sum(j['group'] == 'brec' for j in todo)}")
        return
    # timing alone on the GPU (molhiv excluded): last by default, or first with --timing-first
    tjob = script_job("timing", ["experiments_rebuttal/timing.py", "NCI1", "MUTAG", "PROTEINS", "IMDBBINARY", "ZINC"],
                      os.path.join(EXPR, "timing.done"))
    if TIMING_FIRST:
        run_queue([tjob], 1, {})
    run_queue(jobs, WORKERS, CAPS)
    if not TIMING_FIRST:
        run_queue([tjob], 1, {})
    say("=== all done. Tables: python experiments_rebuttal/paper_tables.py")


if __name__ == "__main__":
    main()
