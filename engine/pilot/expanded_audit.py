"""Direct endpoint audits and complete release/market checks."""
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import t
from .expanded_design import decode,seed,truths
from .expanded_bank import generate,init
from .expanded_fit import ContinuousFitter,basis
from .inference_features import vector
from .observations import PANELS

def endpoints(root,workers=8):
    root=Path(root);dest=root/'endpoint_audit';dest.mkdir(exist_ok=True);jobs=[];info=[]
    for case in truths():
        rows=json.loads((root/'fits'/f'bank_0_O6_truth_{case["truth_id"]:02d}.json').read_text())
        r=next(x for x in rows if x['replicate']==0 and x['starts']==24 and x['participation']=='unknown')
        if r['empty']:continue
        for label in ['lower','upper']:
            i=len(info);x=r[f'{label}_x'];path=dest/f'endpoint_{i:02d}.npz'
            info.append(dict(index=i,truth_id=case['truth_id'],name=case['name'],endpoint=label,x=x,path=path.name))
            jobs.append(dict(path=str(path),parameters=decode(x),role='audit',x=x,stratum=-1,seeds=[seed(0,'audit',i,j) for j in range(80)]))
    (dest/'endpoint_specification.json').write_text(json.dumps(info,indent=2)+'\n')
    pending=[j for j in jobs if not Path(j['path']).exists()]
    init()
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for f in as_completed([pool.submit(generate,j) for j in pending]):
            r=f.result()
            if r['failures']:raise RuntimeError(r)
    model=ContinuousFitter(root/'models/bank_0_O6.npz');result=[]
    for item in info:
        with np.load(dest/item['path']) as z:y=z['O6'];response=z['response']
        x=np.array(item['x']);pred=float(basis(x)@model.response_coef)
        score=np.sum(((y-basis(x)@model.coef)@model.white)**2,axis=1)
        mean=float(response.mean());se=float(response.std(ddof=1)/np.sqrt(len(response)))
        result.append({**item,'n':len(response),'response_mean':mean,'response_mcse':se,'predicted_response':pred,
          'absolute_response_error':abs(pred-mean),'response_error_envelope':model.error,
          'envelope_contains_mean':abs(pred-mean)<=model.error,
          'envelope_contains_99ci':abs(pred-mean)+float(t.ppf(.995,len(response)-1))*se<=model.error,
          'parameter_acceptance_pct':float(100*np.mean(score<=model.cutoff))})
    (dest/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def complete_audit(root):
    root=Path(root);markets=files=0;failures=[];maxclear=maxcash=maxkkt=0.;multiple=0;seen=set()
    folders=[root/f'bank_{b}'/r for b in range(2) for r in ['train','check','calibration','response']]+[root/'evaluation',root/'oracle',root/'endpoint_audit']
    for folder in folders:
        for path in folder.glob('*.npz'):
            with np.load(path) as z:
                if 'metadata' not in z:continue
                m=json.loads(str(z['metadata']));ss=set(map(int,z['seeds']))
                if seen&ss:raise AssertionError(f'Seed collision at {path}')
                seen.update(ss);markets+=m['successful'];files+=1;failures.extend(m['failures'])
                maxclear=max(maxclear,m['max_clearing']);maxcash=max(maxcash,m['max_cash']);maxkkt=max(maxkkt,m['max_kkt']);multiple+=m['multiple_root_periods']
    releases=0;maxdiff=0.;marginaldiff=0.
    for case in truths():
        with np.load(root/'evaluation'/f'truth_{case["truth_id"]:02d}.npz') as z:
            for i,s in enumerate(z['seeds']):
                token=hashlib.sha256(f'release-v03-{s}'.encode()).hexdigest()[:16];summaries={}
                for p in PANELS:
                    with np.load(root/'evaluation/analyst'/f'{token}_{p}.npz') as zz:data={k:zz[k] for k in zz.files}
                    assert not {'seed','truth','response','gamma','parameters','regime'}&set(data)
                    assert data['prices'].shape==(18,) and int(data['months'].max())==18
                    v=vector(data,p);maxdiff=max(maxdiff,float(np.max(np.abs(v-z[p][i]))));releases+=1
                    from .inference_features import summaries as summarize
                    summaries[p]=summarize(data,p)
                marginaldiff=max(marginaldiff,max(abs(summaries['O6'][k]-v) for k,v in summaries['O7'].items()))
    assert not failures and maxdiff<1e-12 and marginaldiff<1e-12
    assert not seen & {s+2_000_000_000 for s in seen}
    result=dict(markets=markets,bank_files=files,failures=len(failures),max_clearing=maxclear,max_cash=maxcash,max_kkt=maxkkt,
      multiple_root_periods=multiple,analyst_releases=releases,max_release_summary_difference=maxdiff,
      max_O6_O7_marginal_difference=marginaldiff,seed_count=len(seen),reserved_confirmation_seeds_used=False)
    (root/'summary/run_audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--complete-only',action='store_true');a=p.parse_args()
    if not a.complete_only:print(json.dumps(endpoints(a.root,a.workers),indent=2))
    print(json.dumps(complete_audit(a.root),indent=2))
