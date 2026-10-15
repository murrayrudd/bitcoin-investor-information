"""Monotone numerical refinement for the holdings-only response projection.

Added after the frozen validation search found disconnected endpoint solutions.
This extension never changes scores, fitted models, cutoffs, or error allowances.
Its reported intervals contain the original intervals by construction.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.optimize import minimize
from .repair_fit import RepairedFitter, basis
from .repair_evaluate import verify_freeze
from .repair_design import cases
from .expanded_evaluate import write_csv, stats
from .inference_features import vector

MODELS={}

def refine(model, y, initial, max_passes=3):
    result=dict(initial)
    if result['empty']:
        result.update(refinement_passes=0, refinement_attempts=0, refinement_failures=0)
        return result
    lo=float(result['lower'])+model.error; hi=float(result['upper'])-model.error
    xlo=np.array(result['lower_x']); xhi=np.array(result['upper_x']); yw=np.asarray(y)@model.white
    cache={}; attempts=failures=0; history=[]
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x, cache['x']):
            v,j=basis(x,True); residual=v@model.wcoef-yw
            cutoff=np.exp(v@model.cutoff_coef); score=residual@residual
            cache.update(x=x.copy(), ratio=float(score/cutoff),
                         ratio_grad=(2*(j@model.wcoef)@residual-score*(j@model.cutoff_coef))/cutoff,
                         response=float(v@model.response_coef), response_grad=j@model.response_coef)
        return cache
    def score(x):
        e=evaluate(x);return e['ratio'],e['ratio_grad']
    def objective(x,sign):
        e=evaluate(x);return sign*e['response'],sign*e['response_grad']
    constraint=dict(type='ineq',fun=lambda x:1-evaluate(x)['ratio'],jac=lambda x:-evaluate(x)['ratio_grad'])
    for sweep in range(max_passes):
        before=(lo,hi); seeds=[]
        for xx,sign in [(xlo.copy(),1),(xhi.copy(),-1)]:
            for d in range(9):
                candidate=xx.copy();candidate[d]=-candidate[d];seeds.append((candidate,sign))
        for candidate,sign in seeds:
            fit=minimize(score,candidate,jac=True,bounds=[(-1.,1.)]*9,method='L-BFGS-B',
                         options=dict(maxiter=240,ftol=1e-11,gtol=1e-7))
            attempts+=1;failures+=int(not fit.success)
            if evaluate(fit.x)['ratio']>1+1e-8:
                continue
            opt=minimize(lambda x:objective(x,sign),fit.x,jac=True,bounds=[(-1.,1.)]*9,
                         constraints=constraint,method='SLSQP',options=dict(maxiter=240,ftol=1e-9))
            attempts+=1;failures+=int(not opt.success)
            for xx in (fit.x,opt.x):
                e=evaluate(xx)
                if e['ratio']<=1+1e-7:
                    if e['response']<lo:lo=e['response'];xlo=xx.copy()
                    if e['response']>hi:hi=e['response'];xhi=xx.copy()
        movement=max(before[0]-lo,hi-before[1]);history.append(float(movement))
        if movement<1e-6:
            break
    result.update(lower=lo-model.error,upper=hi+model.error,width=hi-lo+2*model.error,
                  lower_x=xlo.tolist(),upper_x=xhi.tolist(),refinement_passes=len(history),
                  refinement_attempts=attempts,refinement_failures=failures,refinement_movements=history)
    if result['lower']>initial['lower']+1e-10 or result['upper']<initial['upper']-1e-10:
        raise AssertionError('Refinement narrowed an interval')
    return result

def init(root):
    for bank in (0,1):MODELS[bank]=RepairedFitter(Path(root)/'models'/f'bank_{bank}_O1.npz')

def job(root,bank,case):
    root=Path(root);model=MODELS[bank];tid=case['truth_id'];start=time.perf_counter()
    raw=json.loads((root/'fits'/f'bank_{bank}_O1_truth_{tid:02d}.json').read_text())
    initial=[r for r in raw if r['method']=='v0.4' and r['participation']=='unknown' and r['starts']==16]
    with np.load(root/'evaluation'/f'truth_{tid:02d}.npz',allow_pickle=False) as z:ys=z['O1']
    rows=[]
    for r in initial:
        rep=r['replicate'];answer=refine(model,ys[rep],r);answer['method']='v0.4_refined_O1'
        answer['original_lower']=r['lower'];answer['original_upper']=r['upper'];answer['original_width']=r['width']
        if rep<6:
            large=model.fit(ys[rep],budgets=(96,))[0]
            if large['empty'] or answer['empty']:
                move=None
            else:
                move=max(0.,answer['lower']-large['lower'],large['upper']-answer['upper'])
            answer['comparison_96']=large;answer['remaining_movement_96_pp']=move
        rows.append(answer)
    dest=root/'refined_O1'/f'bank_{bank}_truth_{tid:02d}.json'
    dest.write_text(json.dumps(rows,indent=2)+'\n')
    return dict(bank=bank,truth_id=tid,rows=len(rows),seconds=time.perf_counter()-start)

def summarize(root):
    root=Path(root);rows=[]
    for path in (root/'refined_O1').glob('*.json'):rows+=json.loads(path.read_text())
    if len(rows)!=3840:raise RuntimeError('Incomplete holdings-only refinement')
    import csv
    with open(root/'summary/oracle.csv') as f:
        oracle={int(r['truth_id']):{k:float(r[k]) for k in ('mean','half_99')} for r in csv.DictReader(f)}
    scopes=[];checks=[];cells=[]
    for bank in (0,1):
        selected=[r for r in rows if r['bank']==bank]
        for scope in ('all','previously_seen_setting','new_setting'):
            rr=selected if scope=='all' else [r for r in selected if r['status']==scope]
            scopes.append(dict(bank=bank,panel='O1',scope=scope,**stats(rr,oracle)))
        for case in cases():
            rr=[r for r in selected if r['truth_id']==case['truth_id']]
            cells.append(dict(bank=bank,panel='O1',truth_id=case['truth_id'],name=case['name'],**stats(rr,oracle)))
        checked=[r for r in selected if 'comparison_96' in r]
        movements=[r['remaining_movement_96_pp'] for r in checked if r['remaining_movement_96_pp'] is not None]
        gains=[r['width']-r['original_width'] for r in selected if not r['empty']]
        checks.append(dict(bank=bank,intervals=len(selected),n_compared_with_96=len(checked),
                           max_remaining_movement_pp=float(max(movements)),
                           n_remaining_above_0_005_pp=int(sum(v>.005 for v in movements)),
                           mean_width_increase_pp=float(np.mean(gains)),max_width_increase_pp=float(max(gains)),
                           refinement_optimizer_failures=sum(r['refinement_failures'] for r in selected)))
    write_csv(root/'summary/refined_O1_by_scope.csv',scopes)
    write_csv(root/'summary/refined_O1_by_case.csv',cells)
    write_csv(root/'summary/refined_O1_search_checks.csv',checks)
    print(json.dumps(checks),flush=True)

def run(root,workers=8):
    root=Path(root);verify_freeze(root);(root/'refined_O1').mkdir(exist_ok=True)
    code=Path(__file__);digest=hashlib.sha256(code.read_bytes()).hexdigest()
    frozen=root/'search_refinement_freeze.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['source_sha256']!=digest:raise RuntimeError('Refinement source changed')
    else:
        frozen.write_text(json.dumps(dict(source='pilot/repair_refine.py',source_sha256=digest,
            created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            reason='Three bank 0 holdings-only checks moved by more than 0.005 pp when increasing from 16 to 48 starts.',
            initial_starts=16,endpoint_reflection_passes=3,stop_movement_pp=1e-6,
            additional_check_starts=96,additional_check_first_replicates=6,
            statistical_models_and_calibration_unchanged=True,
            interval_expansion_only=True),indent=2)+'\n')
    pending=[(bank,case) for bank in (0,1) for case in cases()
             if not (root/'refined_O1'/f'bank_{bank}_truth_{case["truth_id"]:02d}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(root),)) as pool:
        futures=[pool.submit(job,str(root),bank,case) for bank,case in pending]
        for i,f in enumerate(as_completed(futures)):
            print(json.dumps(dict(done=i+1,total=len(futures),last=f.result())),flush=True)
    summarize(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,a.workers)
