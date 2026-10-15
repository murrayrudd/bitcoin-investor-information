"""Fresh selected-endpoint checks and complete simulation/release audits."""
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import t
from .expanded_bank import generate, init
from .expanded_design import decode
from .inference_features import vector, summaries
from .observations import PANELS
from .repair_design import cases, seed, COUNTS
from .repair_evaluate import LARGE_STARTS, verify_freeze
from .repair_fit import RepairedFitter, basis

def endpoint_jobs(root):
    root=Path(root); dest=root/'endpoint_audit'; dest.mkdir(exist_ok=True); info=[]; jobs=[]
    crossed={0, 9, 10, 12, 16, 20}
    for case in cases():
        tid=case['truth_id']; combinations=[(0, 'O6')]
        if tid in crossed:
            combinations += [(0, 'O7'), (1, 'O6'), (1, 'O7')]
        for bank, panel in combinations:
            rows=json.loads((root/'fits'/f'bank_{bank}_{panel}_truth_{tid:02d}.json').read_text())
            fit=next(r for r in rows if r['method']=='v0.4' and r['participation']=='unknown'
                     and r['replicate']==0 and r['starts']==LARGE_STARTS[panel])
            if fit['empty']:
                info.append(dict(index=None, bank=bank, panel=panel, truth_id=tid, name=case['name'], empty=True))
                continue
            for side in ('lower', 'upper'):
                i=len(jobs); x=fit[f'{side}_x']; path=dest/f'endpoint_{i:03d}.npz'
                info.append(dict(index=i, bank=bank, panel=panel, truth_id=tid, name=case['name'],
                                 side=side, x=x, path=path.name, empty=False))
                jobs.append(dict(path=str(path), parameters=decode(x), x=x, stratum=-1, role='audit',
                                 seeds=[seed(0, 'audit', i, r) for r in range(COUNTS['endpoint_audit_reps'])]))
    return info, jobs

def endpoints(root, workers=8):
    root=Path(root); verify_freeze(root); info, jobs=endpoint_jobs(root); dest=root/'endpoint_audit'
    (dest/'endpoint_specification.json').write_text(json.dumps(info, indent=2)+'\n')
    pending=[j for j in jobs if not Path(j['path']).exists()]; init()
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for i, future in enumerate(as_completed([pool.submit(generate, j) for j in pending])):
            r=future.result()
            if r['failures']:
                raise RuntimeError(r)
            if i % 8 == 0:
                print(json.dumps(dict(endpoint_jobs_done=i+1, total=len(pending))), flush=True)
    models={}; results=[]
    for item in info:
        if item['empty']:
            results.append(item); continue
        key=(item['bank'], item['panel'])
        if key not in models:
            models[key]=RepairedFitter(root/'models'/f'bank_{key[0]}_{key[1]}.npz')
        model=models[key]; x=np.array(item['x'])
        with np.load(dest/item['path'], allow_pickle=False) as z:
            sample=z['response']; data=z[item['panel']]
            meta=json.loads(str(z['metadata']))
            if meta['failures'] or len(sample) != COUNTS['endpoint_audit_reps']:
                raise RuntimeError('Incomplete endpoint audit')
        pred=float(basis(x)@model.response_coef); mean=float(sample.mean())
        se=float(sample.std(ddof=1)/np.sqrt(len(sample)))
        scores=np.sum(((data-basis(x)@model.coef)@model.white)**2, axis=1)
        cutoff=float(np.exp(basis(x)@model.cutoff_coef))
        results.append(dict(**item, n=len(sample), predicted_response=pred, response_mean=mean, response_mcse=se,
                            response_half_99=float(t.ppf(.995, len(sample)-1)*se),
                            absolute_response_error=abs(pred-mean), allowance_pp=model.error,
                            allowance_contains_mean=bool(abs(pred-mean)<=model.error),
                            allowance_contains_99ci=bool(abs(pred-mean)+t.ppf(.995, len(sample)-1)*se<=model.error),
                            parameter_acceptance_pct=float(100*np.mean(scores<=cutoff)), cutoff=cutoff))
    (dest/'audit_results.json').write_text(json.dumps(results, indent=2)+'\n')
    valid=[r for r in results if not r['empty']]
    summary=dict(endpoints=len(valid), empty_audit_fits=sum(r['empty'] for r in results),
                 n_allowance_contains_mean=sum(r['allowance_contains_mean'] for r in valid),
                 n_allowance_contains_99ci=sum(r['allowance_contains_99ci'] for r in valid),
                 max_response_error_pp=max(r['absolute_response_error'] for r in valid),
                 min_parameter_acceptance_pct=min(r['parameter_acceptance_pct'] for r in valid))
    (dest/'summary.json').write_text(json.dumps(summary, indent=2)+'\n'); return summary

def complete(root):
    root=Path(root); verify_freeze(root); seen=set(); markets=files=multiple=0; clear=cash=kkt=0.
    folders=[root/f'bank_{b}'/role for b in (0, 1) for role in ('train', 'check', 'quantile', 'calibration', 'response')]
    folders += [root/'evaluation', root/'oracle', root/'endpoint_audit']
    for folder in folders:
        for path in folder.glob('*.npz'):
            with np.load(path, allow_pickle=False) as z:
                if 'metadata' not in z:
                    continue
                meta=json.loads(str(z['metadata'])); seeds=set(map(int, z['seeds']))
                if seen & seeds or meta['failures'] or meta['successful'] != meta['attempted']:
                    raise AssertionError(f'Invalid or overlapping simulation bank: {path}')
                if min(seeds)<600_000_000 or max(seeds)>=800_000_000:
                    raise AssertionError('Unexpected seed namespace')
                seen.update(seeds); markets+=meta['successful']; files+=1
                clear=max(clear, meta['max_clearing']); cash=max(cash, meta['max_cash']); kkt=max(kkt, meta['max_kkt'])
                multiple+=meta['multiple_root_periods']
    if seen & {s+2_000_000_000 for s in seen}:
        raise AssertionError('Market/observation seed collision')
    releases=0; difference=marginal=0.
    for case in cases():
        with np.load(root/'evaluation'/f'truth_{case["truth_id"]:02d}.npz', allow_pickle=False) as z:
            for i, s in enumerate(z['seeds']):
                token=hashlib.sha256(f'release-v03-{s}'.encode()).hexdigest()[:16]; ss={}
                for panel in PANELS:
                    with np.load(root/'evaluation/analyst'/f'{token}_{panel}.npz', allow_pickle=False) as zz:
                        data={k:zz[k] for k in zz.files}
                    if {'seed', 'truth', 'response', 'gamma', 'parameters', 'regime'} & set(data):
                        raise AssertionError('Truth leaked into analyst release')
                    if data['prices'].shape != (18,) or int(data['months'].max()) != 18:
                        raise AssertionError('Information cutoff violation')
                    difference=max(difference, float(np.max(np.abs(vector(data, panel)-z[panel][i]))))
                    ss[panel]=summaries(data, panel); releases+=1
                marginal=max(marginal, max(abs(ss['O6'][k]-v) for k, v in ss['O7'].items()))
    if difference>1e-12 or marginal>1e-12:
        raise AssertionError('Release reconstruction or marginal comparison failed')
    result=dict(fresh_markets=markets, simulation_files=files, failures=0, unique_market_seeds=len(seen),
                max_clearing=clear, max_cash=cash, max_kkt=kkt, multiple_root_periods=multiple,
                analyst_releases=releases, max_release_summary_difference=difference,
                max_O6_O7_marginal_difference=marginal, frozen_source_and_models_unchanged=True,
                original_confirmation_seeds_used=False)
    (root/'summary/run_audit.json').write_text(json.dumps(result, indent=2)+'\n'); return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--complete-only', action='store_true'); a=p.parse_args()
    if not a.complete_only:
        print(json.dumps(endpoints(a.root, a.workers)), flush=True)
    print(json.dumps(complete(a.root)), flush=True)
