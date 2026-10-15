"""Checkpointed simulator banks; two entirely independent fitted banks."""
import argparse,json,time,traceback,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import numpy as np
from .config import Config
from .economy import simulate
from .kernel import enable_compiled
from .observations import release_panels,PANELS
from .inference_features import summaries
from .expanded_design import COUNTS,decode,design_points,truths,seed,freeze

def init():enable_compiled(required=True)

def generate(job):
    path=Path(job['path']);start=time.perf_counter();cfg=replace(Config(),**job['parameters']).validate()
    responses=[];features={p:[] for p in PANELS};failures=[];clear=cash=kkt=0.;multiple=0;names={}
    for s in job['seeds']:
        try:
            market=simulate(cfg,s);responses.append(market.response)
            clear=max(clear,max(abs(d['clearing_residual']) for d in market.diagnostics))
            cash=max(cash,max(abs(d['cash_ledger_residual']) for d in market.ledger))
            kkt=max(kkt,max(d['kkt_violation'] for d in market.diagnostics))
            multiple+=sum(len(d['roots'])>1 or bool(d['zero_spans']) for d in market.diagnostics)
            if job['role'] not in ('response','oracle'):
                panels,_=release_panels(market,cfg,s+2_000_000_000,'selective')
                for p,data in panels.items():
                    v=summaries(data,p);features[p].append(list(v.values()));names[p]=list(v)
                    if job['role']=='evaluation':
                        token=hashlib.sha256(f'release-v03-{s}'.encode()).hexdigest()[:16]
                        dest=path.parent/'analyst';dest.mkdir(exist_ok=True)
                        np.savez_compressed(dest/f'{token}_{p}.npz',**data)
        except Exception as exc:failures.append(dict(seed=s,error=repr(exc),traceback=traceback.format_exc()))
    meta=dict(role=job['role'],parameters=job['parameters'],attempted=len(job['seeds']),successful=len(responses),
              failures=failures,max_clearing=clear,max_cash=cash,max_kkt=kkt,multiple_root_periods=multiple,
              feature_names=names,seconds=time.perf_counter()-start)
    arrays=dict(response=np.asarray(responses),seeds=np.asarray(job['seeds']),metadata=np.array(json.dumps(meta)))
    if 'x' in job:arrays['x']=np.asarray(job['x']);arrays['stratum']=np.array(job['stratum'])
    for p,v in features.items():
        if v:arrays[p]=np.asarray(v)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)
    return dict(path=str(path),markets=len(responses),failures=len(failures),seconds=meta['seconds'])

def jobs(out,which):
    out=Path(out)
    if which in ('all','development'):
        for b in range(2):
            for role,n,reps in [('train',COUNTS['train_points'],COUNTS['train_reps']),('check',COUNTS['check_points'],COUNTS['check_reps']),
                                ('calibration',COUNTS['calibration_points'],1),('response',COUNTS['response_points'],COUNTS['response_reps'])]:
                x,strata=design_points(n,700_000+b*100+{'train':1,'check':2,'calibration':3,'response':4}[role],iid=role in ('calibration','response'))
                folder=out/f'bank_{b}'/role;folder.mkdir(parents=True,exist_ok=True)
                for i in range(n):yield dict(path=str(folder/f'{i:04d}.npz'),parameters=decode(x[i]),x=x[i].tolist(),stratum=int(strata[i]),role=role,seeds=[seed(b,role,i,j) for j in range(reps)])
    if which in ('all','validation'):
        for row in truths():
            for role,n in [('evaluation',COUNTS['evaluation_reps']),('oracle',COUNTS['oracle_reps'])]:
                folder=out/role;folder.mkdir(exist_ok=True)
                yield dict(path=str(folder/f'truth_{row["truth_id"]:02d}.npz'),parameters=row['parameters'],role=role,seeds=[seed(0,role,row['truth_id'],j) for j in range(n)])

def run(out,workers=8,which='development'):
    out=Path(out);freeze(out);pending=[];complete=0
    for job in jobs(out,which):
        p=Path(job['path'])
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']))
                if meta['failures'] or meta['successful']!=len(job['seeds']):raise RuntimeError(f'Incomplete {p}')
            complete+=1
        else:pending.append(job)
    enable_compiled(required=True);start=time.perf_counter();logs=[]
    print(json.dumps(dict(jobs=len(pending),existing=complete,markets=sum(len(j['seeds']) for j in pending))),flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        fs=[pool.submit(generate,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            r=f.result();logs.append(r)
            if r['failures']:raise RuntimeError(r)
            progress=dict(done=i+1,total=len(fs),markets=sum(v['markets'] for v in logs),elapsed=time.perf_counter()-start,last=r)
            if i%25==0 or i+1==len(fs):print(json.dumps(progress),flush=True)
            (out/f'progress_{which}.json').write_text(json.dumps(progress,indent=2)+'\n')
    (out/f'jobs_{which}.json').write_text(json.dumps(logs,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--which',choices=['all','development','validation'],default='development');a=p.parse_args();run(a.output,a.workers,a.which)
