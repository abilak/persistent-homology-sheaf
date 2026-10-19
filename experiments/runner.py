"""
python experiments/runner.py core_fast 3 1    (plan, workers, threads each). finished jobs get skipped
python experiments/runner.py all 3            (3 workers, add --dry to just count, --skip=brec_,mp_ to drop stuff)

on the server:
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue python experiments/runner.py all 3 < /dev/null
"""
import os, sys, subprocess, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.environ.get('PY', sys.executable)
RESULTS = os.path.join(ROOT, 'results')

FAST = {'padded_batching': True}
REV = {'topo_multiplicity': True}  # no-op now (counts are always on), kept so result names match
NS_ESS = dict(REV, topo_norm_stats=True, topo_essential=True)
FULL = dict(NS_ESS, topo_filt_squash=True, topo_readout=True)
TOPO = {'rev': REV, 'ns_ess': NS_ESS, 'full': FULL}

TU = ['MUTAG', 'PTC', 'IMDBBINARY', 'PROTEINS', 'NCI1', 'NCI109']
FOLDS = range(1, 11)
S10 = range(10)

# same #params as the full model
WIDE = {'MUTAG': 140, 'PTC': 92, 'NCI1': 138, 'NCI109': 138, 'PROTEINS': 140, 'IMDBBINARY': 140, 'IMDBMULTI': 140, 'ENZYMES': 140, 'ZINC': 138, 'MOLHIV': 132}
MICRO_WIDE = {'MOLHIV': 8, 'PROTEINS': 4}

CAPS = {'proteins': 1, 'brec': 1}
EXPR = os.path.join('results', 'expr')
LOG = None

def say(msg):
    line = '[%s] %s' % (time.strftime('%F %T'), msg)
    print(line, flush=True)
    if(LOG is not None):
        LOG.write(line + '\n')
        LOG.flush()

def job(ds, model, seed, fold, variant, ov):
    return dict(dataset=ds, model=model, seed=seed, fold=fold, epochs=None, variant=variant, overrides=dict(ov))

def tu_jobs(dsets, model, variant, ov, seeds=(0,)):
    jobs = []
    for ds in dsets:
        for s in seeds:
            for f in FOLDS:
                jobs.append(job(ds, model, s, f, variant, ov))
    return jobs

def fix_jobs(ds, model, variant, ov, seeds=S10):
    return [job(ds, model, s, 1, variant, ov) for s in seeds]

def base_full(dsets, seeds=(0,)):
    out = []
    for ds in dsets:
        out += tu_jobs([ds], 'baseline', 'fast', FAST, seeds)
        out += tu_jobs([ds], 'topo', 'full_fast', dict(FULL, **FAST), seeds)
    return out

def fixed_split(ds, vlist, seeds=(0, 1, 2, 3), with_base=True):
    if(with_base):
        out = fix_jobs(ds, 'baseline', 'fast', FAST, seeds)
    else:
        out = []
    for v in vlist:
        topo_ov = dict(TOPO[v], **FAST)
        out += fix_jobs(ds, 'topo', v + '_fast', topo_ov, seeds)
    return out

def wide(dsets, seeds=(0,), fixed=False):
    out = []
    for ds in dsets:
        ov = dict(FAST, block_width=WIDE[ds])
        if(ds in MICRO_WIDE):
            ov['micro_batch'] = MICRO_WIDE[ds]
        if(fixed):
            out += fix_jobs(ds, 'baseline', 'wide_fast', ov, seeds)
        else:
            out += tu_jobs([ds], 'baseline', 'wide_fast', ov, seeds)
    return out

def pair(ds, tag, base_ov, full_ov, fixed=False):
    if(fixed):
        return fix_jobs(ds, 'baseline', tag, base_ov, (0, 1, 2, 3)) + fix_jobs(ds, 'topo', 'full_' + tag, full_ov, (0, 1, 2, 3))
    return tu_jobs([ds], 'baseline', tag, base_ov) + tu_jobs([ds], 'topo', 'full_' + tag, full_ov)

def depth(ds, L, fixed=False):
    w = 32 if ds == 'PTC' else 64
    blocks = [w] * L
    return pair(ds, 'L%d_fast' % L, dict(FAST, block_features=blocks), dict(FULL, block_features=blocks, **FAST), fixed)

PLANS = {}
PLANS['core_fast'] = base_full(TU)
for ds in TU:
    PLANS['core_fast_' + ds.lower()] = base_full([ds])
PLANS['extra_tu_fast'] = base_full(['IMDBMULTI', 'ENZYMES'])
PLANS['imdbb_seeds_fast'] = base_full(['IMDBBINARY'], seeds=(1, 2))
PLANS['zinc_fast_s10'] = fixed_split('ZINC', ['full'], seeds=S10)
PLANS['molhiv_fast_s10'] = fixed_split('MOLHIV', ['full'], seeds=S10)

PLANS['wide_fast'] = wide(['MUTAG', 'PTC', 'IMDBBINARY', 'NCI1', 'NCI109'])
PLANS['wide_fast_proteins'] = wide(['PROTEINS'])
PLANS['wide_extra_tu_fast'] = wide(['IMDBMULTI', 'ENZYMES'])
PLANS['wide_zinc_fast_s10'] = wide(['ZINC'], S10, fixed=True)
PLANS['wide_molhiv_fast_s10'] = wide(['MOLHIV'], S10, fixed=True)

MP = ['mlp', 'gcn', 'gin', 'gsn']
TU8 = ['MUTAG', 'PTC', 'NCI1', 'NCI109', 'PROTEINS', 'ENZYMES', 'IMDBBINARY', 'IMDBMULTI']
mp_jobs = []
for ds in TU8:
    for m in MP:
        for f in FOLDS:
            mp_jobs.append(job(ds, m, 0, f, 'fast', FAST))
PLANS['mp_fast'] = mp_jobs
PLANS['mp_zinc_fast_s10'] = [job('ZINC', m, s, 1, 'fast', FAST) for m in MP for s in S10]
PLANS['mp_molhiv_fast_s10'] = [job('MOLHIV', m, s, 1, 'fast', FAST) for m in MP for s in S10]

STATIC = dict(FULL, topo_static=True, **FAST)
PLANS['static_fast'] = tu_jobs(['MUTAG', 'PTC', 'IMDBBINARY', 'NCI1', 'NCI109'], 'topo', 'full_static_fast', STATIC)
PLANS['static_fast_proteins'] = tu_jobs(['PROTEINS'], 'topo', 'full_static_fast', STATIC)
PLANS['static_extra_tu_fast'] = tu_jobs(['ENZYMES', 'IMDBMULTI'], 'topo', 'full_static_fast', STATIC)
PLANS['static_zinc_fast_s10'] = fix_jobs('ZINC', 'topo', 'full_static_fast', STATIC)

BETTI = dict(FAST, betti_control=True, topo_max_simplex_dim=2)
PLANS['betti_fast'] = tu_jobs(['MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'ENZYMES', 'NCI1', 'NCI109'], 'baseline', 'betti_fast', BETTI)
PLANS['betti_fast_proteins'] = tu_jobs(['PROTEINS'], 'baseline', 'betti_fast', BETTI)
PLANS['betti_zinc_fast_s10'] = fix_jobs('ZINC', 'baseline', 'betti_fast', BETTI)
PLANS['betti_molhiv_fast_s10'] = fix_jobs('MOLHIV', 'baseline', 'betti_fast', BETTI)

STRUCT_BASE = dict(FAST, structure_only=True)
STRUCT_FULL = dict(FULL, structure_only=True, **FAST)
PLANS['structure_fast'] = []
for ds in ['MUTAG', 'PTC', 'ENZYMES', 'NCI1']:
    PLANS['structure_fast'] += pair(ds, 'struct_fast', STRUCT_BASE, STRUCT_FULL)
PLANS['structure_fast_proteins'] = pair('PROTEINS', 'struct_fast', STRUCT_BASE, STRUCT_FULL)

LAST = dict(FULL, topo_apply_layers='last', **FAST)
PLANS['placement_fast'] = tu_jobs(['MUTAG', 'PTC', 'NCI1', 'NCI109'], 'topo', 'full_last_fast', LAST)
PLANS['placement_zinc_fast_s10'] = fix_jobs('ZINC', 'topo', 'full_last_fast', LAST)

for ds in ['MUTAG', 'PTC', 'NCI1']:
    djobs = []
    for L in (1, 3, 4):
        djobs += depth(ds, L)
    PLANS['depth_' + ds.lower()] = djobs
PLANS['depth_zinc'] = depth('ZINC', 1, fixed=True) + depth('ZINC', 3, fixed=True) + depth('ZINC', 4, fixed=True)

ablation = []
for ds in ['MUTAG', 'PTC', 'NCI1', 'NCI109']:
    for v in ('rev', 'ns_ess'):
        ablation += tu_jobs([ds], 'topo', v + '_fast', dict(TOPO[v], **FAST))
PLANS['ablation_min_fast'] = ablation
PLANS['zinc_ablation_fast'] = fixed_split('ZINC', ['rev', 'ns_ess'], with_base=False)
PLANS['molhiv_ablation_fast'] = fixed_split('MOLHIV', ['rev', 'ns_ess'], with_base=False)

smoke = []
for f in (1, 2):
    smoke.append(dict(job('MUTAG', 'baseline', 0, f, 'fast', FAST), epochs=3))
for f in (1, 2):
    smoke.append(dict(job('MUTAG', 'topo', 0, f, 'full_fast', dict(FULL, **FAST)), epochs=3))
smoke.append(dict(job('NCI1', 'topo', 0, 1, 'full_fast', dict(FULL, **FAST)), epochs=2))
PLANS['smoke'] = smoke

MAIN = ['core_fast', 'extra_tu_fast', 'zinc_fast_s10', 'molhiv_fast_s10', 'wide_fast', 'wide_extra_tu_fast', 'wide_zinc_fast_s10', 'wide_molhiv_fast_s10', 'wide_fast_proteins', 'mp_fast', 'mp_zinc_fast_s10', 'mp_molhiv_fast_s10', 'static_fast', 'static_zinc_fast_s10', 'static_fast_proteins', 'static_extra_tu_fast', 'betti_fast', 'betti_zinc_fast_s10', 'betti_molhiv_fast_s10', 'betti_fast_proteins', 'imdbb_seeds_fast']
STUDIES = ['structure_fast', 'structure_fast_proteins', 'placement_fast', 'placement_zinc_fast_s10', 'ablation_min_fast', 'zinc_ablation_fast', 'molhiv_ablation_fast', 'depth_mutag', 'depth_zinc', 'depth_ptc', 'depth_nci1']

def rpath(j):
    ep = '_e%d' % j['epochs'] if j.get('epochs') else ''
    var = '_' + j['variant'] if j.get('variant') else ''
    fname = '%s_s%d_f%d%s%s.json' % (j['model'], j['seed'], j['fold'], ep, var)
    return os.path.join(RESULTS, j['dataset'], fname)

def job_cmd(j):
    cmd = [PY, os.path.join(HERE, 'run_job.py'), j['dataset'], j['model'], str(j['seed']), str(j['fold']), rpath(j)]
    if(j.get('overrides')):
        if(j.get('epochs')):
            cmd.append(str(j['epochs']))
        else:
            cmd.append('-')
        cmd.append(json.dumps(j['overrides']))
    elif(j.get('epochs')):
        cmd.append(str(j['epochs']))
    return cmd

def plan_jobs(names):
    jobs = []
    seen = set()
    for name in names:
        for j in PLANS[name]:
            rp = rpath(j)
            if(rp in seen):
                continue
            seen.add(rp)
            group = 'proteins' if j['dataset'] == 'PROTEINS' else 'light'
            jobs.append(dict(name='%s:%s/%s/s%d/f%d' % (name, j['dataset'], j['model'], j['seed'], j['fold']), cmd=job_cmd(j), done=rp, log=rp + '.log', group=group))
    return jobs

def script(name, cmd, done, group='light'):
    return dict(name=name, cmd=[PY] + cmd, done=done, log=os.path.join(EXPR, name + '.log'), group=group)

def expr_jobs():
    jobs = [script('interpret_mutag', ['experiments/interpret_mutag.py'], 'results/interpret_mutag/stats.json')]
    for s in (1, 2):
        jobs.append(script('interpret_mutag_seed%d' % s, ['experiments/interpret_mutag.py', '--seed=%d' % s], 'results/interpret_mutag_seed%d/stats.json' % s))
    jobs.append(script('srg_analysis', ['experiments/srg.py'], EXPR + '/srg_results.json'))
    jobs.append(script('srg_seeds', ['experiments/srg.py', '--seeds', '30'], EXPR + '/srg_seeds_results.json'))
    jobs.append(script('witness_cone', ['experiments/srg.py', 'cone', '--seeds=30'], EXPR + '/witness_cone.json'))
    cone_out = EXPR + '/sr_classification_cone.json'
    jobs.append(script('cone_classification', ['experiments/sr_classification.py', 'CONE', '--D=2', '--P=1', '--seeds=3', '--models=gin,gsn,baseline,topo', '--out=' + cone_out], cone_out))
    jobs.append(script('count_vs_ph', ['experiments/srg.py', 'count'], EXPR + '/count_vs_ph_results.json'))
    jobs.append(script('sr_classification', ['experiments/sr_classification.py', '--seeds=3'], EXPR + '/sr_classification.json'))
    for m in ('topo', 'ppgn', 'gin'):
        out = 'results/brec/%s.json' % m
        brec_cmd = ['experiments/brec.py', '--model', m, '--data', 'data/brec_v3.npy', '--sample-num', '400', '--epochs', '20', '--out', out]
        jobs.append(script('brec_' + m, brec_cmd, out, group='brec'))
    return jobs

def run_plan(name, workers, threads):
    todo = [j for j in PLANS[name] if not os.path.exists(rpath(j))]
    print('[%s] %d jobs, %d left, %d workers x %d threads' % (name, len(PLANS[name]), len(todo), workers, threads))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    running = []
    done = 0
    while(todo or running):
        while(todo and len(running) < workers):
            j = todo.pop(0)
            rp = rpath(j)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            log = open(rp + '.log', 'w')
            p = subprocess.Popen(job_cmd(j), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, j, log, time.time()))
        time.sleep(2)
        still = []
        for(p, j, log, t0) in running:
            if(p.poll() is None):
                still.append((p, j, log, t0))
                continue
            log.close()
            done += 1
            ok = p.returncode == 0 and os.path.exists(rpath(j))
            status = 'ok' if ok else 'FAIL rc=%s' % p.returncode
            print('[%d] %s %s/%s/s%d/f%d (%ds)' % (done, status, j['dataset'], j['model'], j['seed'], j['fold'], time.time() - t0), flush=True)
        running = still
    print('[%s] done' % name)

def run_queue(jobs, workers, caps):
    todo = [j for j in jobs if not os.path.exists(j['done'])]
    say('%d jobs, %d to run, %d workers, caps %s' % (len(jobs), len(todo), workers, caps))
    running = []
    n_done = 0
    retried = set()
    t_start = time.time()
    while(todo or running):
        started = True
        while(started and len(running) < workers):
            started = False
            active = {}
            for r in running:
                g = r['job']['group']
                active[g] = active.get(g, 0) + 1
            for(k, j) in enumerate(todo):
                if(active.get(j['group'], 0) >= caps.get(j['group'], workers)):
                    continue
                todo.pop(k)
                os.makedirs(os.path.dirname(j['log']) or '.', exist_ok=True)
                lf = open(j['log'], 'w')
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                p = subprocess.Popen(j['cmd'], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
                running.append(dict(p=p, job=j, lf=lf, t0=time.time()))
                started = True
                break
        time.sleep(5)
        still = []
        for r in running:
            if(r['p'].poll() is None):
                still.append(r)
                continue
            r['lf'].close()
            j = r['job']
            n_done += 1
            ok = r['p'].returncode == 0 and os.path.exists(j['done'])
            if(not ok and j['name'] not in retried):
                retried.add(j['name'])
                todo.append(j)
                tag = 'FAIL (retrying later)'
            elif(ok):
                tag = 'ok'
            else:
                tag = 'FAIL'
            mins = (time.time() - r['t0']) / 60
            hours = (time.time() - t_start) / 3600
            say('[%d] %s %s (%.1f min) | %d left, %.1f h in | %s' % (n_done, tag, j['name'], mins, len(todo) + len(still), hours, j['log']))
        running = still

def everything():
    expr = expr_jobs()
    slow = ('sr_classification', 'brec_')
    fast_expr = [j for j in expr if not j['name'].startswith(slow)]
    slow_expr = [j for j in expr if j['name'].startswith(slow)]
    return fast_expr + plan_jobs(MAIN) + plan_jobs(STUDIES) + slow_expr

def run_all(argv):
    global LOG
    os.chdir(ROOT)
    args = [a for a in argv if not a.startswith('--')]
    workers = int(args[0]) if args else 3
    dry = '--dry' in argv
    skip = []
    for a in argv:
        if(a.startswith('--skip=')):
            for s in a[len('--skip='):].split(','):
                if(s):
                    skip.append(s)
    skip = tuple(skip)
    os.makedirs(EXPR, exist_ok=True)
    LOG = open(os.path.join('results', 'run_everything.log'), 'a')

    say('start')
    jobs = everything()
    if(skip):
        jobs = [j for j in jobs if not j['name'].split(':')[0].startswith(skip)]
        say(f'skipping {skip}')
    if(dry):
        left = {}
        for j in jobs:
            if(not os.path.exists(j['done'])):
                k = j['name'].split(':')[0]
                left[k] = left.get(k, 0) + 1
        say(f'dry run: {sum(left.values())} of {len(jobs)} left: {left}')
        sys.exit(0)
    run_queue(jobs, workers, CAPS)
    run_queue([script('timing', ['experiments/timing.py', 'NCI1', 'MUTAG', 'PROTEINS', 'IMDBBINARY', 'ZINC', 'MOLHIV'], 'results/timing.json')], 1, {})
    say('all done, tables: python experiments/tables.py')

if(__name__ == '__main__'):
    if(len(sys.argv) > 1 and sys.argv[1] == 'all'):
        run_all(sys.argv[2:])
    elif(len(sys.argv) < 2 or sys.argv[1] not in PLANS):
        print('plans: all', ' '.join(PLANS))
        sys.exit(1)
    else:
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        run_plan(sys.argv[1], workers, threads)
