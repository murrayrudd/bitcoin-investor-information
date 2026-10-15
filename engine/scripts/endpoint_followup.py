"""Independent diagnostic repeat of the three lowest-retention audit locations.

This is a selected follow-up; it does not change any fitted model or interval.
"""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import binomtest,t
from pilot.repair_design import seed
from pilot.expanded_design import decode
from pilot.expanded_bank import generate,init
from pilot.repair_fit import RepairedFitter,basis
from pilot.repair_evaluate import verify_freeze
from pilot.expanded_evaluate import wilson

def run(root,workers=4):
    root=Path(root);verify_freeze(root);dest=root/'endpoint_followup';dest.mkdir(exist_ok=True)
    rows=json.loads((root/'endpoint_audit/audit_results.json').read_text())
    selected=sorted([r for r in rows if not r['empty']],key=lambda r:r['parameter_acceptance_pct'])[:3]
    spec=dict(selection_rule='Three smallest observed acceptance rates in the 84-endpoint audit; ties follow original index order.',
              replications=512,original_indices=[r['index'] for r in selected],models_unchanged=True,
              calibration_updated=False,selection_is_diagnostic=True)
    (dest/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    jobs=[]
    for i,r in enumerate(selected):
        jobs.append(dict(path=str(dest/f'followup_{i:02d}.npz'),parameters=decode(r['x']),x=r['x'],
                         stratum=-1,role='audit',seeds=[seed(0,'construction',i,j) for j in range(512)]))
    init()
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for future in as_completed([pool.submit(generate,j) for j in jobs if not Path(j['path']).exists()]):
            if future.result()['failures']:raise RuntimeError('Follow-up simulation failure')
    out=[]
    for i,r in enumerate(selected):
        m=RepairedFitter(root/'models'/f'bank_{r["bank"]}_{r["panel"]}.npz');x=np.array(r['x'])
        with np.load(dest/f'followup_{i:02d}.npz') as z:ys=z[r['panel']];response=z['response']
        scores=np.sum(((ys-basis(x)@m.coef)@m.white)**2,axis=1)
        cutoff=float(np.exp(basis(x)@m.cutoff_coef));accepted=int(np.sum(scores<=cutoff));n=len(scores)
        lo,hi=wilson(accepted,n);p=float(binomtest(accepted,n,.965,alternative='less').pvalue)
        mean=float(response.mean());se=float(response.std(ddof=1)/np.sqrt(n));pred=float(basis(x)@m.response_coef)
        out.append(dict(original_index=r['index'],bank=r['bank'],panel=r['panel'],name=r['name'],side=r['side'],
                        previous_acceptance_pct=r['parameter_acceptance_pct'],n=n,accepted=accepted,
                        acceptance_pct=100*accepted/n,wilson_low_pct=100*lo,wilson_high_pct=100*hi,
                        one_sided_p_under_96_5_pct=p,bonferroni_p_three=min(1.,3*p),
                        response_mean=mean,response_mcse=se,response_abs_error=abs(pred-mean),
                        allowance_contains_99ci=bool(abs(pred-mean)+t.ppf(.995,n-1)*se<=m.error)))
    (dest/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/repaired_pilot');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.root,a.workers)
