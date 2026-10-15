"""Evaluate frozen fits against a separate response oracle and paired panels."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.stats import t
from .inference import FitModel
from .inference_design import WIDTH_THRESHOLD
from .run import save_csv,save_json


def wilson(success,n,z=1.959963984540054):
    phat=success/n;den=1+z*z/n
    center=(phat+z*z/(2*n))/den
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return center-half,center+half


def summarize(rows,keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for group,rr in groups.items():
        n=len(rr);coverage=np.array([r['covered'] for r in rr]);empty=np.array([r['empty'] for r in rr]);width=np.array([r['width'] for r in rr if not r['empty']])
        bcover=np.array([r['bootstrap_covered'] for r in rr]);errors=np.array([r['point_error'] for r in rr])
        lo,hi=wilson(int(coverage.sum()),n)
        result.append(dict(zip(keys,group),n=n,covered=int(coverage.sum()),coverage=float(coverage.mean()),coverage_mcse=float(np.sqrt(coverage.mean()*(1-coverage.mean())/n)),coverage_wilson_low=lo,coverage_wilson_high=hi,
          coverage_definite=float(np.mean([r['coverage_definite'] for r in rr])),coverage_possible=float(np.mean([r['coverage_possible'] for r in rr])),
          empty_count=int(empty.sum()),empty_rate=float(empty.mean()),mean_width=None if not len(width) else float(width.mean()),median_width=None if not len(width) else float(np.median(width)),
          mean_retained=float(np.mean([r['retained_count'] for r in rr])),single_family_rate=float(np.mean([r['family_count']==1 for r in rr])),
          point_bias=float(errors.mean()),point_rmse=float(np.sqrt(np.mean(errors**2))),
          misleading_precision_rate=float(np.mean([r['misleading_precision'] for r in rr])),
          bootstrap_coverage=float(bcover.mean()),bootstrap_mean_width=float(np.mean([r['bootstrap_width'] for r in rr])),bootstrap_misleading_precision_rate=float(np.mean([r['bootstrap_misleading_precision'] for r in rr]))))
    return result


def run(output):
    out=Path(output);dest=out/'summary';dest.mkdir(exist_ok=True)
    spec=json.loads((out/'inference_specification.json').read_text());tic=time.perf_counter()
    oracle={};oracle_rows=[]
    evaluation={}
    for truth in spec['truth_cases']:
        i=truth['truth_id']
        with np.load(out/'oracle'/f'truth_{i:02d}.npz') as z:
            meta=json.loads(str(z['metadata']))
            if meta['failures']:raise ValueError('Oracle bank contains failures')
            r=z['oracle_response'];mean=float(r.mean());se=float(r.std(ddof=1)/np.sqrt(len(r)))
            half=float(t.ppf(.995,len(r)-1)*se)
            oracle[i]=(mean,mean-half,mean+half)
            oracle_rows.append(dict(truth_id=i,name=truth['name'],kind=truth['kind'],n=len(r),mean=mean,mcse=se,lower_99=mean-half,upper_99=mean+half))
        with np.load(out/'evaluation'/f'truth_{i:02d}.npz') as z:
            meta=json.loads(str(z['metadata']))
            if meta['failures']:raise ValueError('Evaluation bank contains failures; do not silently drop cases')
            evaluation[i]={k:z[k] for k in z.files if k.startswith('evaluation_')}
    save_csv(dest/'oracle_means.csv',oracle_rows)
    cases=[('clean_correct','clean','clean'),('selective_correct','selective','selective'),('selective_ignored','selective','clean')]
    rows=[];model_diagnostics=[]
    with (dest/'retained_sets.jsonl').open('w') as detailed:
        for panel in ('O1','O2','O6','O7'):
            for assumption in ('clean','selective'):
                model=FitModel.load(out,panel,assumption)
                model_diagnostics.append(dict(panel=panel,assumption=assumption,features=model.means.shape[1],precision_condition=float(np.linalg.cond(model.precision)),reference_halfwidth_min=float(model.response_halfwidth.min()),reference_halfwidth_max=float(model.response_halfwidth.max())))
                for case,actual,assumed in cases:
                    if assumed!=assumption:continue
                    for truth in spec['truth_cases']:
                        i=truth['truth_id'];mean,oracle_low,oracle_high=oracle[i]
                        data=evaluation[i];features=data[f'evaluation_{actual}_{panel}']
                        for rep,feature in enumerate(features):
                            seed=int(data['evaluation_seeds'][rep])
                            token=hashlib.sha256(f'release-v02-{seed}-{actual}'.encode()).hexdigest()[:16]
                            for grid in ('primary','refined'):
                                fit=model.fit(feature,grid)
                                empty=fit['lower'] is None;lo=fit['lower'];hi=fit['upper']
                                covered=not empty and lo<=mean<=hi
                                blo=fit['bootstrap_lower'];bhi=fit['bootstrap_upper'];bcov=blo<=mean<=bhi
                                record=dict(dataset_id=f'{token}_{panel}',truth_id=i,truth_name=truth['name'],truth_kind=truth['kind'],truth_family=truth['family'],replication=rep,case=case,oracle=mean,**fit)
                                detailed.write(json.dumps(record,allow_nan=False)+'\n')
                                rows.append(dict(dataset_id=record['dataset_id'],truth_id=i,truth_name=truth['name'],truth_kind=truth['kind'],replication=rep,case=case,panel=panel,grid=grid,
                                  oracle=mean,lower=lo,upper=hi,width=fit['width'],empty=int(empty),covered=int(covered),
                                  coverage_definite=int(not empty and lo<=oracle_low and hi>=oracle_high),coverage_possible=int(not empty and lo<=oracle_high and hi>=oracle_low),
                                  retained_count=len(fit['retained']),family_count=len(fit['retained_families']),families=';'.join(fit['retained_families']),
                                  best=fit['best'],best_family=fit['best_family'],point=fit['point'],point_error=fit['point']-mean,
                                  bootstrap_lower=blo,bootstrap_upper=bhi,bootstrap_width=fit['bootstrap_width'],bootstrap_covered=int(bcov),
                                  misleading_precision=int(not empty and not covered and fit['width']<WIDTH_THRESHOLD),bootstrap_misleading_precision=int(not bcov and fit['bootstrap_width']<WIDTH_THRESHOLD),maximum_pvalue=fit['maximum_pvalue']))
                print(f'Fitted {panel}/{assumption}; {len(rows)} records',flush=True)
    save_csv(dest/'fits.csv',rows)
    save_csv(dest/'by_truth.csv',summarize(rows,['truth_name','truth_kind','case','panel','grid']))
    pooled=summarize(rows,['truth_kind','case','panel','grid']);save_csv(dest/'by_scope.csv',pooled)
    # Paired width differences only when both intervals exist; always retain
    # paired coverage and empty-set differences over all evaluation economies.
    pairs=defaultdict(dict)
    for r in rows:
        if r['panel'] in ('O6','O7'):pairs[(r['truth_id'],r['replication'],r['case'],r['grid'])][r['panel']]=r
    paired_rows=[]
    for key,pair in pairs.items():
        a,b=pair['O6'],pair['O7']
        paired_rows.append(dict(truth_kind=a['truth_kind'],case=a['case'],grid=a['grid'],coverage_difference=a['covered']-b['covered'],empty_difference=a['empty']-b['empty'],width_difference=None if a['empty'] or b['empty'] else a['width']-b['width']))
    groups=defaultdict(list)
    for r in paired_rows:groups[(r['truth_kind'],r['case'],r['grid'])].append(r)
    paired_summary=[]
    for group,rr in groups.items():
        a=np.array([r['coverage_difference'] for r in rr],float);w=np.array([r['width_difference'] for r in rr if r['width_difference'] is not None])
        paired_summary.append(dict(zip(['truth_kind','case','grid'],group),n=len(rr),coverage_difference=float(a.mean()),coverage_difference_mcse=float(a.std(ddof=1)/np.sqrt(len(a))),empty_difference=float(np.mean([r['empty_difference'] for r in rr])),width_pairs=len(w),width_difference=None if len(w)==0 else float(w.mean()),width_difference_mcse=None if len(w)<2 else float(w.std(ddof=1)/np.sqrt(len(w)))))
    save_csv(dest/'linked_vs_unlinked.csv',paired_summary)
    # Nested catalog test: fixed scores/calibration imply retained sets and
    # response hulls can only expand under refinement (apart from emptiness).
    matched=defaultdict(dict)
    for r in rows:matched[(r['dataset_id'],r['case'])][r['grid']]=r
    for pair in matched.values():
        a,b=pair['primary'],pair['refined']
        if not a['empty']:
            if b['empty'] or b['lower']>a['lower']+1e-12 or b['upper']<a['upper']-1e-12:raise AssertionError('Refinement lost an accepted response')
    allmeta=[]
    for folder in ('candidate_banks','evaluation','oracle'):
        for path in sorted((out/folder).glob('*.npz')):
            with np.load(path) as z:meta=json.loads(str(z['metadata']))
            allmeta.append(meta)
    diagnostics=[d for m in allmeta for d in m['diagnostics']]
    save_json(dest/'run_audit.json',dict(version=spec['version'],candidate_count=len(spec['catalog']),primary_count=sum(r['tier']=='primary' for r in spec['catalog']),truth_count=len(spec['truth_cases']),evaluation_markets=sum(len(d['evaluation_seeds']) for d in evaluation.values()),fitted_records=len(rows),released_datasets=len(list((out/'evaluation'/'analyst').glob('*.npz'))),total_markets=sum(d['attempted'] for d in diagnostics),failed_markets=sum(len(m['failures']) for m in allmeta),max_clearing=max(d['max_clearing'] for d in diagnostics),max_cash=max(d['max_cash'] for d in diagnostics),max_kkt=max(d['max_kkt'] for d in diagnostics),multiple_root_periods=sum(d['multiple_root_periods'] for d in diagnostics),nested_grid_checks=len(matched),fit_seconds=time.perf_counter()-tic,simulation_worker_seconds=sum(m['seconds'] for m in allmeta),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,model_diagnostics=model_diagnostics))
    print(json.dumps(pooled,indent=2),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
