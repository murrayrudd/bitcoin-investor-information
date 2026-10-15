"""Generate independent checkpointed banks. No fitting uses evaluation truth."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .config import Config
from .economy import simulate
from .kernel import enable_compiled
from .observations import release_panels,PANELS,REGIMES
from .inference_features import summaries
from .inference_design import *


def _init():enable_compiled(required=True)


def generate_job(job):
    path=Path(job['path']);tic=time.perf_counter()
    cfg=replace(Config(),**job['parameters']).validate()
    arrays={};failures=[];diagnostics=[];feature_names={}
    for role,seeds in job['banks'].items():
        responses=[];features={f'{r}_{p}':[] for r in REGIMES for p in PANELS}
        max_clear=max_cash=max_kkt=0.0;multiple=0
        for seed in seeds:
            try:
                market=simulate(cfg,int(seed))
                responses.append(market.response)
                max_clear=max(max_clear,max(abs(d['clearing_residual']) for d in market.diagnostics))
                max_cash=max(max_cash,max(abs(d['cash_ledger_residual']) for d in market.ledger))
                max_kkt=max(max_kkt,max(d['kkt_violation'] for d in market.diagnostics))
                multiple+=sum(len(d['roots'])>1 or bool(d['zero_spans']) for d in market.diagnostics)
                if role in ('train','calibration','evaluation'):
                    for regime in REGIMES:
                        panels,_=release_panels(market,cfg,int(seed)+1_000_000_000,regime)
                        for panel,data in panels.items():
                            summary=summaries(data,panel);feature_names[panel]=list(summary)
                            features[f'{regime}_{panel}'].append(list(summary.values()))
                            if job.get('store_releases'):
                                token=hashlib.sha256(f'release-v02-{seed}-{regime}'.encode()).hexdigest()[:16]
                                dest=path.parent/'analyst';dest.mkdir(exist_ok=True)
                                np.savez_compressed(dest/f'{token}_{panel}.npz',**data)
            except Exception as exc:
                failures.append(dict(role=role,seed=int(seed),error=repr(exc),traceback=traceback.format_exc()))
        arrays[f'{role}_seeds']=np.asarray(seeds,dtype=np.int64)
        arrays[f'{role}_response']=np.asarray(responses)
        for key,values in features.items():
            if values:arrays[f'{role}_{key}']=np.asarray(values,float)
        diagnostics.append(dict(role=role,attempted=len(seeds),successful=len(responses),max_clearing=max_clear,max_cash=max_cash,max_kkt=max_kkt,multiple_root_periods=multiple))
    metadata=dict(parameters=job['parameters'],seconds=time.perf_counter()-tic,diagnostics=diagnostics,failures=failures,feature_names=feature_names)
    arrays['metadata']=np.array(json.dumps(metadata,allow_nan=False))
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)
    return dict(path=path.name,seconds=metadata['seconds'],failures=len(failures),markets=sum(d['attempted'] for d in diagnostics))


def run(output,workers=8,which='all'):
    out=Path(output);spec=freeze(out)
    for name in ('candidate_banks','evaluation','oracle'):(out/name).mkdir(exist_ok=True)
    jobs=[]
    if which in ('all','candidates'):
        for row in spec['catalog']:
            i=row['candidate_id'];path=out/'candidate_banks'/f'candidate_{i:03d}.npz'
            jobs.append(dict(path=str(path),parameters=row['parameters'],banks={role:seed_bank(role,i,n).tolist() for role,n in [('train',N_TRAIN),('calibration',N_CAL),('reference',N_REF)]}))
    if which in ('all','evaluation','oracle'):
        for row in spec['truth_cases']:
            i=row['truth_id']
            for role,n in [('evaluation',N_EVAL),('oracle',N_ORACLE)]:
                if which not in ('all',role):continue
                path=out/role/f'truth_{i:02d}.npz'
                jobs.append(dict(path=str(path),parameters=row['parameters'],banks={role:seed_bank(role,i,n).tolist()},store_releases=role=='evaluation'))
    pending=[]
    for job in jobs:
        path=Path(job['path'])
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']))
                if meta['failures']:raise RuntimeError(f'Incomplete bank {path}')
            continue
        pending.append(job)
    enable_compiled(required=True)
    print(f'{len(pending)} jobs pending; {len(jobs)-len(pending)} complete checkpoints; workers={workers}',flush=True)
    start=time.perf_counter();logs=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_init) as pool:
        fs=[pool.submit(generate_job,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            result=f.result();logs.append(result)
            print(json.dumps(dict(done=i+1,total=len(fs),elapsed=round(time.perf_counter()-start,1),**result)),flush=True)
            (out/f'bank_progress_{which}.json').write_text(json.dumps(logs,indent=2)+'\n')
    return logs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--which',choices=['all','candidates','evaluation','oracle'],default='all')
    a=p.parse_args();run(a.output,a.workers,a.which)
