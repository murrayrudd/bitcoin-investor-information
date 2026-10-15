"""Descriptive comparisons; never updates fitted models or intervals."""
import json
from pathlib import Path
import numpy as np
from pilot.repair_design import cases
from pilot.expanded_design import encode
from pilot.expanded_fit import ContinuousFitter, basis as old_basis
from pilot.repair_fit import RepairedFitter, basis
from pilot.expanded_evaluate import write_csv
from pilot.observations import PANELS
from pilot.repair_evaluate import MAIN_STARTS

def run(root, previous):
    root=Path(root);previous=Path(previous);dest=root/'summary';dest.mkdir(exist_ok=True)
    out=[]
    for bank in (0,1):
        old=ContinuousFitter(previous/'models'/f'bank_{bank}_O6.npz')
        new=RepairedFitter(root/'models'/f'bank_{bank}_O6.npz')
        for case in cases():
            tid=case['truth_id'];x=encode(case['parameters'])
            with np.load(root/'oracle'/f'truth_{tid:02d}.npz') as z:sample=z['response']
            mean=float(sample.mean());se=float(sample.std(ddof=1)/np.sqrt(len(sample)))
            op=float(old_basis(x)@old.response_coef);npred=float(basis(x)@new.response_coef)
            out.append(dict(bank=bank,truth_id=tid,name=case['name'],oracle_mean=mean,oracle_mcse=se,
                            old_prediction=op,new_prediction=npred,old_abs_error=abs(op-mean),new_abs_error=abs(npred-mean)))
    write_csv(dest/'response_approximation_by_case.csv',out)
    errors=[]
    for bank in (0,1):
        rows=[r for r in out if r['bank']==bank]
        errors.append(dict(bank=bank,old_rmse_pp=float(np.sqrt(np.mean([r['old_abs_error']**2 for r in rows]))),
                           new_rmse_pp=float(np.sqrt(np.mean([r['new_abs_error']**2 for r in rows]))),
                           old_max_abs_error_pp=max(r['old_abs_error'] for r in rows),
                           new_max_abs_error_pp=max(r['new_abs_error'] for r in rows)))
    write_csv(dest/'response_approximation_summary.csv',errors)
    rows=[]
    for path in (root/'fits').glob('*.json'):rows+=json.loads(path.read_text())
    optimizer=[]
    for bank in (0,1):
        for panel in PANELS:
            selected=[r for r in rows if r['bank']==bank and r['panel']==panel and r['method']=='v0.4'
                      and r['participation']=='unknown' and r['starts']==MAIN_STARTS[panel]]
            optimizer.append(dict(bank=bank,panel=panel,intervals=len(selected),
                                   attempts=sum(r['counts']['score_minimizations']+r['counts']['endpoint_optimizations'] for r in selected),
                                   unsuccessful_statuses=sum(r['counts']['optimizer_failures'] for r in selected)))
    write_csv(dest/'optimizer_statuses.csv',optimizer)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/repaired_pilot');p.add_argument('--previous',default='results/expanded_pilot')
    a=p.parse_args();run(a.root,a.previous)
