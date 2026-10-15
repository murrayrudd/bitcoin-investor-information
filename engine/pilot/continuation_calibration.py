"""Simultaneous finite-anchor order-statistic calibration, independent of validation."""
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import binom
from .expanded_bank import generate,init
from .repair_fit import RepairedFitter,basis
from .continuation_design import ROOT,PREVIOUS,PANELS,COUNTS,anchors,seed,save_spec

def tolerance_rank(n, content=.965, failure=.05/96):
    for k in range(1,n+1):
        if binom.cdf(k-1,n,content)>=1-failure:return k
    raise ValueError('Sample too small for finite tolerance limit')

def run(root=ROOT,previous=PREVIOUS,workers=8):
    root=Path(root);previous=Path(previous);spec=save_spec(root)
    dest=root/'calibration';dest.mkdir(exist_ok=True);jobs=[]
    for r in spec['anchors']:
        path=dest/f'bank_{r["bank"]}_anchor_{r["index"]:02d}.npz'
        jobs.append(dict(path=str(path),parameters=r['parameters'],x=r['x'],stratum=-1,role='calibration',
           seeds=[seed(r['bank'],'calibration',r['index'],j) for j in range(COUNTS['calibration_reps'])]))
    pending=[]
    for j in jobs:
        p=Path(j['path'])
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                m=json.loads(str(z['metadata']))
                if m['failures'] or not np.array_equal(z['seeds'],j['seeds']):raise RuntimeError('Invalid checkpoint')
        else:pending.append(j)
    init();t=time.perf_counter()
    print(json.dumps(dict(stage='calibration',jobs=len(pending),fresh_markets=sum(len(j['seeds']) for j in pending))),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        fs=[pool.submit(generate,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            r=f.result()
            if r['failures']:raise RuntimeError(r)
            print(json.dumps(dict(done=i+1,total=len(fs),seconds=time.perf_counter()-t,last=r)),flush=True)
    models=root/'models';models.mkdir(exist_ok=True);details=[];summaries=[]
    k=tolerance_rank(COUNTS['calibration_reps'])
    for bank in (0,1):
        for panel in PANELS:
            oldpath=previous/'models'/f'bank_{bank}_{panel}.npz';m=RepairedFitter(oldpath);limits=[]
            for a in [r for r in spec['anchors'] if r['bank']==bank]:
                with np.load(dest/f'bank_{bank}_anchor_{a["index"]:02d}.npz',allow_pickle=False) as z:
                    y=z[panel];x=z['x']
                b=basis(x);scores=np.sum(((y-b@m.coef)@m.white)**2,axis=1)
                oldcut=float(np.exp(b@m.cutoff_coef));ratios=scores/oldcut;limit=float(np.sort(ratios)[k-1]);limits.append(limit)
                details.append(dict(bank=bank,panel=panel,anchor=a['index'],old_audit_index=a['old_audit_index'],n=len(y),rank=k,
                    tolerance_ratio=limit,old_retention_pct=100*float(np.mean(ratios<=1))))
            factor=max(1.,max(limits))
            with np.load(oldpath,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
            meta=json.loads(str(arrays['metadata']));meta.update(version='0.5.0',parent_version='0.4.0',
                cutoff_multiplier=factor,tolerance_rank=k,tolerance_n=COUNTS['calibration_reps'],
                tolerance_family_size=spec['tolerance_family_size'],tolerance_family_confidence=.95,
                coverage_scope=spec['scope'],calibration_anchor_indices=[a['old_audit_index'] for a in spec['anchors'] if a['bank']==bank])
            arrays['cutoff_coef']=arrays['cutoff_coef'].copy();arrays['cutoff_coef'][0]+=np.log(factor)
            arrays['metadata']=np.array(json.dumps(meta))
            np.savez_compressed(models/f'bank_{bank}_{panel}.npz',**arrays)
            summaries.append(dict(bank=bank,panel=panel,multiplier=factor,rank=k,n=COUNTS['calibration_reps'],max_anchor_ratio=max(limits)))
    (root/'calibration_results.json').write_text(json.dumps(dict(models=summaries,anchors=details),indent=2)+'\n')
    print(json.dumps(dict(stage='calibration_complete',models=summaries)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();run(a.root,workers=a.workers)
