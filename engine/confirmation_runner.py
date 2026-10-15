"""Frozen prospective Monte Carlo confirmation using unchanged v0.5.0 inference.

The driver handles allocation and durable checkpoints. Economic simulation,
observation mappings, summaries, and all inference use the archived modules.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import t

from pilot.config import Config
from pilot.economy import simulate
from pilot.expanded_design import BOUNDS, PARAMETERS, encode, decode
from pilot.kernel import enable_compiled
from pilot.observations import release_panels
from pilot.inference_features import vector
from pilot.continuation_fit import DirectedFitter

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/confirmation_v06'
PARENT = BASE/'results/continuation_v05'
PANELS = ('O1','O2','O6','O7')
COMPONENTS = ('mixed','belief','funding','allowance','faces','corners')
NSET = (144,16,16,16,32,32)
MAIN = dict(O1=64,O2=24,O6=24,O7=24)
LARGE = dict(O1=128,O2=48,O6=48,O7=48)
SEED_BASE = 330_000_000_000
N_EVAL, N_REF, N_AUDIT = 64,1024,128
MODELS = {}

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
def read_json(path): return json.loads(Path(path).read_text())
def seed(role,index,rep=0):
    roles=dict(evaluation=0,reference=1,audit=2,parameters=3,search=4,bootstrap=5)
    if role not in roles or not 0<=index<1000 or not 0<=rep<10000: raise ValueError('Seed allocation')
    return SEED_BASE+roles[role]*10_000_000+index*10000+rep
def token(s): return hashlib.sha256(f'confirmation-v06-{s}'.encode()).hexdigest()[:24]

def design():
    cases=[];zero=encode(dict(belief_drift=0.,funding_rate=0.,allowance_growth=0.))
    for h,(component,n) in enumerate(zip(COMPONENTS,NSET)):
        rng=np.random.default_rng(seed('parameters',h))
        xx=rng.uniform(-1.,1.,(n,9))
        if component in ('belief','funding','allowance'):
            active={'belief':5,'funding':6,'allowance':7}[component]
            for j in (5,6,7):
                if j!=active: xx[:,j]=zero[j]
        elif component=='faces':
            for x in xx:
                ids=rng.choice(9,int(rng.integers(1,5)),replace=False)
                x[ids]=rng.choice([-1.,1.],len(ids))*rng.uniform(.8,1.,len(ids))
        elif component=='corners':
            xx=rng.choice([-1.,1.],(n,9))*rng.uniform(.8,1.,(n,9))
        for j,x in enumerate(xx):
            cases.append(dict(case_id=len(cases),component=component,component_index=h,within_component=j,
                primary=h<4,weight=[.75,1/12,1/12,1/12,0.,0.][h],x=x.tolist(),parameters=decode(x),
                endpoint_audit=j<2))
    return cases

def planned_seeds():
    arrays=[np.array([seed(role,i,r) for i in range(n) for r in range(reps)],dtype=np.int64)
            for role,n,reps in [('evaluation',256,N_EVAL),('reference',256,N_REF),('audit',144,N_AUDIT)]]
    return np.concatenate(arrays)

def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'source_and_model_freeze.json').exists(): verify();return
    if any((ROOT/f).exists() for f in ('evaluation','reference','fits','audit')):
        raise RuntimeError('Cannot freeze after confirmation output exists')
    # Verify all archived model and economic/inference sources before copying.
    parent_freeze=read_json(PARENT/'source_and_model_freeze.json')
    verified=[]
    for name,expected in parent_freeze['sha256'].items():
        p=BASE/name
        if digest(p)!=expected: raise RuntimeError('Changed calibrated input '+name)
        verified.append(name)
    models=ROOT/'models';models.mkdir(exist_ok=True)
    for src in sorted((PARENT/'models').glob('*.npz')): shutil.copy2(src,models/src.name)
    if len(list(models.glob('*.npz')))!=8: raise RuntimeError('Expected eight frozen models')
    cases=design();old_cases=read_json(PARENT/'specification.json')['cases']
    oldx=np.array([encode(c['parameters']) for c in old_cases])
    for case in cases:
        replace(Config(),**case['parameters']).validate()
        if np.any(np.max(np.abs(oldx-np.array(case['x'])),axis=1)<1e-12):
            raise RuntimeError('New setting duplicates a previous validation setting')
    planned=planned_seeds()
    if len(planned)!=296960 or len(np.unique(planned))!=len(planned): raise RuntimeError('New seed collision')
    # Read actual saved market seeds, rather than assuming namespace separation.
    old_seeds=set();scanned=0
    for p in (BASE/'results').rglob('*.npz'):
        if ROOT in p.parents: continue
        with np.load(p,allow_pickle=False) as z:
            if 'seeds' in z.files:
                old_seeds.update(int(v) for v in z['seeds'].ravel());scanned+=1
    old_with_observations=old_seeds|{s+2_000_000_000 for s in old_seeds}
    proposed=set(int(s) for s in planned)
    new_obs={seed('evaluation',i,r)+2_000_000_000 for i in range(256) for r in range(N_EVAL)}
    new_obs|={seed('audit',i,r)+2_000_000_000 for i in range(144) for r in range(N_AUDIT)}
    if proposed & old_with_observations or new_obs & old_with_observations or proposed & new_obs:
        raise RuntimeError('Seed reuse across stages')
    spec=dict(experiment_version='0.6.0',model_version='0.5.0',created_utc=now(),
        status='authorized prospective confirmation; no main outcomes generated at freeze',
        config=asdict(Config()),parameters=PARAMETERS,bounds=BOUNDS.tolist(),cases=cases,
        components=list(COMPONENTS),component_counts=list(NSET),primary_weights=[.75,1/12,1/12,1/12],
        counts=dict(settings=256,evaluation_per_setting=N_EVAL,reference_per_setting=N_REF,
            evaluation_markets=16384,reference_markets=262144,primary_intervals=131072,
            endpoint_locations_max=144,endpoint_reps=N_AUDIT,endpoint_markets_max=18432,
            total_economic_simulations_max=296960,bootstrap_resamples=2000),
        panels=list(PANELS),banks=[0,1],main_budgets=MAIN,large_budgets=LARGE,
        seed_base=SEED_BASE,seed_rule='330000000000 + role*10000000 + index*10000 + replicate; roles evaluation=0 reference=1 audit=2 parameters=3 search=4 bootstrap=5; observation seed=market seed+2000000000',
        reserved_root=330000,confirmation_root_extended_to_64_bit_namespace=True,
        audit_cases=[c['case_id'] for c in cases if c['endpoint_audit']],audit_panels=['O1','O6','O7'],
        main_search_checks=dict(double_budget_replicates=[0,1],O1_DE_replicate=0,DE_population=96,DE_iterations=160,
            all_empty_intervals='additional doubled-budget and DE diagnostics; preserve primary results',tolerance_pp=.005),
        contrasts=[['O1','O2'],['O1','O6'],['O7','O6']],
        inference_claim='Average performance and paired information differences under the declared design allocations; face/corner stress samples separate; not uniform pointwise coverage.',
        stopping_rule='Fixed allocation. Simulation failures pause for diagnosis without substitute draws. Inference/search/audit failures remain results and do not cause tuning or replacement of primary intervals.',
        freeze_policy='Economic source, released summaries, calibrated model files and primary inference are unchanged from v0.5.0.',
        preliminary_seeds_scanned_files=scanned,preliminary_distinct_market_seeds=len(old_seeds),
        seed_overlap=0,calibrated_inputs_verified=len(verified),
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform()))
    atomic_json(ROOT/'specification.json',spec)
    paths=sorted((BASE/'pilot').glob('*.py'))+[BASE/'pilot/_excess.cpp',Path(__file__),BASE/'confirmation_summary.py',BASE/'confirmation_checks.py',ROOT/'specification.json']
    paths+=sorted(models.glob('*.npz'))
    hashes={str(p.relative_to(BASE)):digest(p) for p in paths}
    atomic_json(ROOT/'source_and_model_freeze.json',dict(created_utc=now(),sha256=hashes,confirmation_not_yet_generated=True))
    print(json.dumps(dict(stage='frozen',settings=256,seed_overlap=0,inputs=len(hashes),time=now())),flush=True)

def verify():
    frozen=read_json(ROOT/'source_and_model_freeze.json')
    for name,value in frozen['sha256'].items():
        if digest(BASE/name)!=value: raise RuntimeError('Frozen input changed '+name)

def simulation_init(): enable_compiled(required=True)

def simulation_job(job):
    path=Path(job['path']);path.parent.mkdir(parents=True,exist_ok=True)
    cfg=replace(Config(),**job['parameters']).validate();role=job['role'];start=time.perf_counter()
    ys={p:[] for p in PANELS};responses=[];later=[];failures=[];seeds=[];clear=cash=kkt=0.;multiple=0
    for s in job['seeds']:
        try:
            market=simulate(cfg,int(s));responses.append(market.response);later.append(market.response_six)
            clear=max(clear,max(abs(d['clearing_residual']) for d in market.diagnostics))
            cash=max(cash,max(abs(d['cash_ledger_residual']) for d in market.ledger))
            kkt=max(kkt,max(d['kkt_violation'] for d in market.diagnostics))
            multiple+=sum(len(d['roots'])>1 or bool(d['zero_spans']) for d in market.diagnostics)
            if cash>1e-8 or clear>1e-8: raise RuntimeError('Confirmation accounting acceptance residual exceeded')
            if role!='reference':
                panels,_=release_panels(market,cfg,int(s)+2_000_000_000,'selective')
                for panel,data in panels.items():
                    value=vector(data,panel);ys[panel].append(value)
                    if role=='evaluation':
                        dest=ROOT/'analyst'/f'{token(s)}_{panel}.npz';dest.parent.mkdir(exist_ok=True)
                        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(dest)
                        with np.load(dest,allow_pickle=False) as z:replay=vector({k:z[k] for k in z.files},panel)
                        np.testing.assert_array_equal(value,replay)
            seeds.append(int(s))
        except Exception as exc:
            failures.append(dict(seed=int(s),error=repr(exc),traceback=traceback.format_exc()));break
    meta=dict(role=role,case_id=job.get('case_id'),parameters=job['parameters'],attempted=len(job['seeds']),
        successful=len(seeds),failures=failures,max_clearing=clear,max_cash=cash,max_kkt=kkt,
        multiple_root_periods=int(multiple),seconds=time.perf_counter()-start)
    arr=dict(seeds=np.asarray(seeds,dtype=np.int64),response=np.asarray(responses[:len(seeds)]),
        response24=np.asarray(later[:len(seeds)]),metadata=np.array(json.dumps(meta)))
    for p,y in ys.items():
        if y:arr[p]=np.asarray(y[:len(seeds)])
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arr);tmp.replace(path)
    if failures: raise RuntimeError(json.dumps(dict(path=str(path),failures=failures)))
    return dict(kind='simulation',role=role,case_id=job.get('case_id'),markets=len(seeds),path=str(path),seconds=meta['seconds'])

def complete_sim(job):
    p=Path(job['path'])
    if not p.exists():return False
    with np.load(p,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']))
        if m['failures'] or m['successful']!=len(job['seeds']) or not np.array_equal(z['seeds'],job['seeds']):
            raise RuntimeError('Incomplete simulation checkpoint '+str(p))
    if job['role']=='evaluation':
        for s in job['seeds']:
            for panel in PANELS:
                if not (ROOT/'analyst'/f'{token(s)}_{panel}.npz').exists():return False
    return True

def simulation_jobs():
    for c in read_json(ROOT/'specification.json')['cases']:
        i=c['case_id']
        for role,n,chunk in [('evaluation',N_EVAL,16),('reference',N_REF,128)]:
            for first in range(0,n,chunk):
                yield dict(path=str(ROOT/role/f'case_{i:03d}_{first:04d}.npz'),parameters=c['parameters'],
                    role=role,case_id=i,seeds=[seed(role,i,r) for r in range(first,min(first+chunk,n))])

def fit_init():
    for bank in (0,1):
        for panel in PANELS: MODELS[bank,panel]=DirectedFitter(ROOT/'models'/f'bank_{bank}_{panel}.npz')

def fit_path(bank,panel,case_id,first):return ROOT/'fits'/f'bank_{bank}_{panel}_case_{case_id:03d}_{first:02d}.json'

def fit_one(bank,panel,case_id,first):
    m=MODELS[bank,panel];rows=[];start=time.perf_counter()
    for rep in range(first,min(first+4,N_EVAL)):
        s=seed('evaluation',case_id,rep)
        rawpath=ROOT/'analyst'/f'{token(s)}_{panel}.npz'
        with np.load(rawpath,allow_pickle=False) as z:y=vector({k:z[k] for k in z.files},panel)
        budgets=(MAIN[panel],LARGE[panel]) if rep<2 else (MAIN[panel],)
        fits=m.fit(y,budgets=budgets)
        common=dict(bank=bank,panel=panel,case_id=case_id,replicate=rep,release_id=rawpath.stem)
        for f in fits:rows.append(dict(**common,**f,method='primary' if f['starts']==MAIN[panel] else 'larger_budget'))
        if fits[0]['empty'] and len(fits)==1:
            extra=m.fit(y,budgets=(MAIN[panel],LARGE[panel]))
            if extra[0]!=fits[0]: raise RuntimeError('Primary search not deterministic')
            rows.append(dict(**common,**extra[-1],method='empty_larger_budget'))
        if (panel=='O1' and rep==0) or fits[0]['empty']:
            check_seed=seed('search',case_id,bank*1000+PANELS.index(panel)*200+rep*2)
            g=m.global_check(y,fits[0],seed=check_seed)
            rows.append(dict(**common,**g,method='DE_check' if panel=='O1' and rep==0 else 'empty_DE_check'))
    dest=fit_path(bank,panel,case_id,first);atomic_json(dest,rows)
    return dict(kind='fit',bank=bank,panel=panel,case_id=case_id,first=first,primary_fits=4,rows=len(rows),seconds=time.perf_counter()-start)

def complete_fit(bank,panel,case_id,first):
    p=fit_path(bank,panel,case_id,first)
    if not p.exists():return False
    rows=read_json(p);primary=[r for r in rows if r['method']=='primary']
    if len(primary)!=4 or {r['replicate'] for r in primary}!=set(range(first,first+4)):raise RuntimeError('Incomplete fit checkpoint')
    for r in primary:
        if r['bank']!=bank or r['panel']!=panel or r['case_id']!=case_id:raise RuntimeError('Misassigned fit checkpoint')
    return True

def audit_jobs():
    endpoints=[];jobs=[]
    spec=read_json(ROOT/'specification.json')
    for case_id in spec['audit_cases']:
        for bank in (0,1):
            for panel in ('O1','O6','O7'):
                rows=read_json(fit_path(bank,panel,case_id,0))
                row=next(r for r in rows if r['method']=='primary' and r['replicate']==0)
                for side in ('lower','upper'):
                    index=len(endpoints)
                    e=dict(index=index,case_id=case_id,bank=bank,panel=panel,side=side,missing=row['empty'])
                    if not row['empty']:
                        x=row[side+'_x'];e.update(x=x,allowance=row['response_error_pp'],
                            predicted_response=row[side]+(1 if side=='lower' else -1)*row['response_error_pp'])
                        jobs.append(dict(path=str(ROOT/'audit'/f'endpoint_{index:03d}.npz'),parameters=decode(x),
                            case_id=case_id,role='audit',seeds=[seed('audit',index,r) for r in range(N_AUDIT)]))
                    endpoints.append(e)
    p=ROOT/'audit/selected_endpoints.json'
    if p.exists() and read_json(p)!=endpoints:raise RuntimeError('Audit selection changed')
    atomic_json(p,endpoints)
    return jobs

def run_pool(stage,jobs,worker,initializer,workers):
    start=time.perf_counter();last_print=0.;done=0;markets=0
    atomic_json(ROOT/f'progress_{stage}.json',dict(stage=stage,status='running',pending_jobs=len(jobs),completed_this_session=0,started_utc=now()))
    print(json.dumps(dict(stage=stage,pending_jobs=len(jobs),workers=workers,time=now())),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=initializer) as pool:
        future=[pool.submit(worker,*j) if isinstance(j,tuple) else pool.submit(worker,j) for j in jobs]
        try:
            for f in as_completed(future):
                result=f.result();done+=1;markets+=result.get('markets',0);elapsed=time.perf_counter()-start
                progress=dict(stage=stage,status='running',pending_jobs=len(jobs),completed_this_session=done,
                    markets_this_session=markets,elapsed_seconds=elapsed,updated_utc=now(),last=result)
                atomic_json(ROOT/f'progress_{stage}.json',progress)
                if elapsed-last_print>30 or done==len(jobs):print(json.dumps(progress),flush=True);last_print=elapsed
        except BaseException as exc:
            for f in future:f.cancel()
            atomic_json(ROOT/f'failure_{stage}.json',dict(time=now(),error=repr(exc),traceback=traceback.format_exc()))
            raise
    atomic_json(ROOT/f'completed_{stage}.json',dict(stage=stage,completed_utc=now(),jobs_this_session=done,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(dict(stage=stage,status='complete',jobs=done,elapsed_seconds=time.perf_counter()-start,time=now())),flush=True)

def run(stage,workers=8):
    verify();enable_compiled(required=True)
    if stage=='generate':
        jobs=[j for j in simulation_jobs() if not complete_sim(j)]
        run_pool(stage,jobs,simulation_job,simulation_init,workers)
    elif stage=='fit':
        if not (ROOT/'completed_generate.json').exists():raise RuntimeError('Generate first')
        jobs=[(bank,panel,i,first) for i in range(256) for bank in (0,1) for panel in PANELS for first in range(0,N_EVAL,4)
              if not complete_fit(bank,panel,i,first)]
        run_pool(stage,jobs,fit_one,fit_init,workers)
    elif stage=='audit':
        if not (ROOT/'completed_fit.json').exists():raise RuntimeError('Fit first')
        jobs=[j for j in audit_jobs() if not complete_sim(j)]
        run_pool(stage,jobs,simulation_job,simulation_init,workers)
    else:raise ValueError(stage)
    verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','generate','fit','audit','all']);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.stage=='freeze':initialize()
    elif a.stage=='all':
        for stage in ('generate','fit','audit'):run(stage,a.workers)
    else:run(a.stage,a.workers)
