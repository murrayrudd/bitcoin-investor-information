"""Frozen fresh validation, isolated-release fitting, and selected endpoint audits."""
import argparse,hashlib,json,time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .expanded_bank import generate,init as simulator_init
from .expanded_design import encode,decode
from .continuation_design import ROOT,PREVIOUS,PANELS,COUNTS,MAIN_BUDGETS,LARGE_BUDGETS,seed,save_spec
from .continuation_fit import DirectedFitter
from .inference_features import vector

MODELS={}

def freeze(root=ROOT):
    root=Path(root);save_spec(root);base=Path(__file__).resolve().parent.parent
    sources=list((base/'pilot').glob('*.py'))+[base/'pilot/_excess.cpp']
    paths=sources+list((root/'models').glob('*.npz'))+[root/'specification.json']
    if len(list((root/'models').glob('*.npz')))!=8:raise RuntimeError('Calibration incomplete')
    hashes={str(p.resolve().relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target=root/'source_and_model_freeze.json'
    if target.exists():
        if json.loads(target.read_text())['sha256']!=hashes:raise RuntimeError('Frozen inputs changed')
    else:target.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,
                 validation_not_yet_generated=True,original_confirmation_seeds_used=False),indent=2)+'\n')

def verify(root=ROOT):
    root=Path(root);base=Path(__file__).resolve().parent.parent
    frozen=json.loads((root/'source_and_model_freeze.json').read_text())
    for p,digest in frozen['sha256'].items():
        if hashlib.sha256((base/p).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen input changed: '+p)

def complete(job):
    p=Path(job['path'])
    if not p.exists():return False
    with np.load(p,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']))
        if m['failures'] or m['successful']!=len(job['seeds']) or not np.array_equal(z['seeds'],job['seeds']):
            raise RuntimeError('Invalid checkpoint '+str(p))
    return True

def generate_validation(root=ROOT,workers=8):
    root=Path(root);verify(root);spec=json.loads((root/'specification.json').read_text());jobs=[]
    for case in spec['cases']:
        for role in ('validation','oracle'):
            folder=root/role;folder.mkdir(exist_ok=True);i=case['truth_id'];n=COUNTS[role+'_reps']
            jobs.append(dict(path=str(folder/f'case_{i:02d}.npz'),parameters=case['parameters'],
               role='audit' if role=='validation' else 'oracle',x=encode(case['parameters']).tolist(),stratum=-1,
               seeds=[seed(0,role,i,r) for r in range(n)]))
    pending=[j for j in jobs if not complete(j)];simulator_init();start=time.perf_counter()
    print(json.dumps(dict(stage='validation',jobs=len(pending),fresh_markets=sum(len(j['seeds']) for j in pending))),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=simulator_init) as pool:
        fs=[pool.submit(generate,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            r=f.result()
            if r['failures']:raise RuntimeError(r)
            print(json.dumps(dict(stage='validation',done=i+1,total=len(fs),seconds=time.perf_counter()-start,last=r)),flush=True)
    # Reproduce only the fitted subset as protected raw releases. These are
    # deterministic replays of existing markets, not additional independent draws.
    folder=root/'released';folder.mkdir(exist_ok=True);jobs=[]
    for case in spec['cases']:
        i=case['truth_id'];jobs.append(dict(path=str(folder/f'case_{i:02d}.npz'),parameters=case['parameters'],role='evaluation',
                seeds=[seed(0,'validation',i,r) for r in range(COUNTS['fitted_reps'])]))
    with ProcessPoolExecutor(max_workers=workers,initializer=simulator_init) as pool:
        fs=[pool.submit(generate,j) for j in jobs if not complete(j)]
        for f in as_completed(fs):
            r=f.result()
            if r['failures']:raise RuntimeError(r)
    checks=0
    for case in spec['cases']:
        tid=case['truth_id']
        with np.load(root/'validation'/f'case_{tid:02d}.npz',allow_pickle=False) as z:
            for rep in range(COUNTS['fitted_reps']):
                token=hashlib.sha256(f'release-v03-{int(z["seeds"][rep])}'.encode()).hexdigest()[:16]
                for panel in PANELS:
                    with np.load(folder/'analyst'/f'{token}_{panel}.npz',allow_pickle=False) as raw:
                        y=vector({k:raw[k] for k in raw.files},panel)
                    np.testing.assert_array_equal(y,z[panel][rep]);checks+=1
    (root/'release_reproduction.json').write_text(json.dumps(dict(exact_reproductions=checks,all_passed=True),indent=2)+'\n')
    print(json.dumps(dict(stage='validation_generation_complete',release_reproductions=checks)),flush=True)

def fit_init(root):
    for bank in (0,1):
        for panel in PANELS:MODELS[bank,panel]=DirectedFitter(Path(root)/'models'/f'bank_{bank}_{panel}.npz')

def fit_job(root,bank,panel,case):
    root=Path(root);tid=case['truth_id'];m=MODELS[bank,panel];rows=[];start=time.perf_counter()
    for rep in range(COUNTS['fitted_reps']):
        s=seed(0,'validation',tid,rep);token=hashlib.sha256(f'release-v03-{s}'.encode()).hexdigest()[:16]
        with np.load(root/'released/analyst'/f'{token}_{panel}.npz',allow_pickle=False) as raw:
            y=vector({k:raw[k] for k in raw.files},panel)
        budgets=(MAIN_BUDGETS[panel],LARGE_BUDGETS[panel]) if rep<2 else (MAIN_BUDGETS[panel],)
        fits=m.fit(y,budgets=budgets)
        common=dict(bank=bank,panel=panel,truth_id=tid,name=case['name'],status=case['status'],replicate=rep,seed=s)
        for f in fits:rows.append(dict(**common,**f,method='v0.5'))
        if panel=='O1' and rep==0:
            g=m.global_check(y,fits[0],seed=955000+bank*100+tid)
            rows.append(dict(**common,**g,method='v0.5_de_check'))
    dest=root/'fits'/f'bank_{bank}_{panel}_case_{tid:02d}.json';temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');temp.replace(dest)
    return dict(bank=bank,panel=panel,case=tid,rows=len(rows),seconds=time.perf_counter()-start)

def fit_all(root=ROOT,workers=8):
    root=Path(root);verify(root);spec=json.loads((root/'specification.json').read_text());(root/'fits').mkdir(exist_ok=True)
    jobs=[(bank,panel,c) for bank in (0,1) for panel in PANELS for c in spec['cases']
          if not (root/'fits'/f'bank_{bank}_{panel}_case_{c["truth_id"]:02d}.json').exists()]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=fit_init,initargs=(str(root),)) as pool:
        fs=[pool.submit(fit_job,str(root),*j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            result=f.result();progress=dict(stage='fitting',done=i+1,total=len(fs),seconds=time.perf_counter()-start,last=result)
            print(json.dumps(progress),flush=True);(root/'fit_progress.json').write_text(json.dumps(progress,indent=2)+'\n')

def endpoint_audit(root=ROOT,workers=8):
    root=Path(root);verify(root);spec=json.loads((root/'specification.json').read_text());dest=root/'endpoint_audit';dest.mkdir(exist_ok=True)
    endpoints=[];jobs=[]
    for bank in (0,1):
        for panel in spec['endpoint_audit_panels']:
            for tid in spec['endpoint_audit_cases']:
                fits=json.loads((root/'fits'/f'bank_{bank}_{panel}_case_{tid:02d}.json').read_text())
                fit=next(r for r in fits if r['replicate']==0 and r['method']=='v0.5' and r['starts']==MAIN_BUDGETS[panel])
                for side in ('lower','upper'):
                    if fit['empty']:raise RuntimeError('Prespecified audit endpoint is empty')
                    i=len(endpoints);x=fit[side+'_x']
                    endpoints.append(dict(index=i,bank=bank,panel=panel,truth_id=tid,name=spec['cases'][tid]['name'],side=side,x=x,
                         predicted_response=fit[side]+(1 if side=='lower' else -1)*fit['response_error_pp'],allowance=fit['response_error_pp']))
                    jobs.append(dict(path=str(dest/f'endpoint_{i:03d}.npz'),parameters=decode(x),x=x,stratum=-1,role='audit',
                              seeds=[seed(0,'audit',i,r) for r in range(COUNTS['audit_reps'])]))
    target=dest/'selected_endpoints.json';content=json.dumps(endpoints,indent=2)+'\n'
    if target.exists() and target.read_text()!=content:raise RuntimeError('Selected endpoints changed')
    target.write_text(content);pending=[j for j in jobs if not complete(j)];simulator_init();start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=simulator_init) as pool:
        fs=[pool.submit(generate,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            r=f.result()
            if r['failures']:raise RuntimeError(r)
            if i%8==0 or i+1==len(fs):print(json.dumps(dict(stage='endpoint_audit',done=i+1,total=len(fs),seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','generate','fit','audit']);p.add_argument('--root',default=str(ROOT));p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='freeze':freeze(a.root)
    elif a.stage=='generate':generate_validation(a.root,a.workers)
    elif a.stage=='fit':fit_all(a.root,a.workers)
    else:endpoint_audit(a.root,a.workers)
