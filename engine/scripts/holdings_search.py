"""Larger common search budget for O1; fixed statistical models and nested sets.

The first reflection experiment did not consistently match the larger search.
The delivered O1 benchmark therefore uses 96 ordinary starts in every market.
"""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pilot.repair_fit import RepairedFitter
from pilot.repair_design import cases
from pilot.repair_evaluate import verify_freeze
from pilot.expanded_evaluate import write_csv,stats

MODELS={}
def init(root):
    for b in (0,1):MODELS[b]=RepairedFitter(Path(root)/'models'/f'bank_{b}_O1.npz')

def job(root,b,case):
    root=Path(root);tid=case['truth_id'];m=MODELS[b]
    baseline=json.loads((root/'fits'/f'bank_{b}_O1_truth_{tid:02d}.json').read_text())
    initial={r['replicate']:r for r in baseline if r['method']=='v0.4' and r['participation']=='unknown' and r['starts']==16}
    with np.load(root/'evaluation'/f'truth_{tid:02d}.npz') as z:ys=z['O1']
    rows=[]
    for rep,y in enumerate(ys):
        budgets=(16,48,96,192) if rep<6 else (16,96)
        for result in m.fit(y,budgets=budgets):
            old=initial[rep]
            if not old['empty'] and (result['empty'] or result['lower']>old['lower']+1e-8 or result['upper']<old['upper']-1e-8):
                raise AssertionError('Extended search failed monotonicity')
            common={k:old[k] for k in ('bank','panel','truth_id','name','kind','status','replicate','seed',
                                     'true_parameter_accepted','true_score','true_cutoff','old_true_parameter_accepted')}
            rows.append(dict(**common,method='v0.4_O1_extended_search',participation='unknown',**result))
    path=root/'holdings_search'/f'bank_{b}_truth_{tid:02d}.json'
    path.write_text(json.dumps(rows,indent=2)+'\n')
    return dict(bank=b,truth_id=tid,rows=len(rows))

def summarize(root):
    import csv
    root=Path(root);rows=[]
    for path in (root/'holdings_search').glob('*.json'):rows+=json.loads(path.read_text())
    with open(root/'summary/oracle.csv') as f:
        oracle={int(r['truth_id']):{k:float(r[k]) for k in ('mean','half_99')} for r in csv.DictReader(f)}
    main=[r for r in rows if r['starts']==96]
    if len(main)!=3840:raise RuntimeError('Incomplete extended O1 search')
    scopes=[];cells=[];checks=[]
    for b in (0,1):
        selected=[r for r in main if r['bank']==b]
        for scope in ('all','previously_seen_setting','new_setting'):
            rr=selected if scope=='all' else [r for r in selected if r['status']==scope]
            scopes.append(dict(bank=b,panel='O1',scope=scope,**stats(rr,oracle)))
        for case in cases():
            rr=[r for r in selected if r['truth_id']==case['truth_id']]
            cells.append(dict(bank=b,panel='O1',truth_id=case['truth_id'],name=case['name'],**stats(rr,oracle)))
        index={(r['truth_id'],r['replicate'],r['starts']):r for r in rows if r['bank']==b}
        for small,large in ((16,96),(96,192)):
            movements=[];gains=[];n=0;new=0
            for (tid,rep,starts),big in index.items():
                if starts!=large:continue
                little=index[tid,rep,small];n+=1;new+=int(little['empty'] and not big['empty'])
                if not little['empty'] and not big['empty']:
                    movements.append(max(little['lower']-big['lower'],big['upper']-little['upper']))
                    gains.append(big['width']-little['width'])
            checks.append(dict(bank=b,small=small,large=large,n=n,new_nonempty=new,
                max_endpoint_movement_pp=float(max(movements)),p95_endpoint_movement_pp=float(np.quantile(movements,.95)),
                n_above_0_005_pp=int(sum(v>.005 for v in movements)),mean_width_increase_pp=float(np.mean(gains))))
    write_csv(root/'summary/holdings_search_by_scope.csv',scopes)
    write_csv(root/'summary/holdings_search_by_case.csv',cells)
    write_csv(root/'summary/holdings_search_checks.csv',checks)
    print(json.dumps(checks),flush=True)

def run(root,workers=8):
    root=Path(root);verify_freeze(root);(root/'holdings_search').mkdir(exist_ok=True)
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();path=root/'holdings_search_freeze.json'
    if path.exists():
        if json.loads(path.read_text())['source_sha256']!=digest:raise RuntimeError('Search extension changed')
    else:
        path.write_text(json.dumps(dict(source='scripts/holdings_search.py',source_sha256=digest,
            created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            main_starts=96,comparison_starts=192,comparison_first_replicates=6,
            statistical_models_unchanged=True,nested_expansion_only=True,
            reason='The 16-start and reflection searches missed solutions revealed by 96 starts. The final benchmark applies 96 starts uniformly.'),indent=2)+'\n')
    pending=[(b,c) for b in (0,1) for c in cases() if not (root/'holdings_search'/f'bank_{b}_truth_{c["truth_id"]:02d}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(root),)) as pool:
        fs=[pool.submit(job,str(root),b,c) for b,c in pending]
        for i,f in enumerate(as_completed(fs)):print(json.dumps(dict(done=i+1,total=len(fs),last=f.result())),flush=True)
    summarize(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/repaired_pilot');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,a.workers)
