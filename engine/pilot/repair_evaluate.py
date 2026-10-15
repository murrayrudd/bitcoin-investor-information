"""Independent fixed-case diagnostics and paired observation comparisons."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import t
from .observations import PANELS
from .expanded_design import encode
from .expanded_fit import ContinuousFitter, basis as old_basis
from .expanded_evaluate import write_csv, wilson, stats
from .repair_fit import RepairedFitter
from .repair_design import cases

MODELS={}; OLD={}
MAIN_STARTS={'O1':16, 'O2':12, 'O6':12, 'O7':12}
LARGE_STARTS={'O1':48, 'O2':32, 'O6':32, 'O7':32}

def init(root, previous):
    for bank in (0, 1):
        for panel in PANELS:
            MODELS[bank, panel]=RepairedFitter(Path(root)/'models'/f'bank_{bank}_{panel}.npz')
            OLD[bank, panel]=ContinuousFitter(Path(previous)/'models'/f'bank_{bank}_{panel}.npz')

def freeze(root):
    root=Path(root); sources=Path(__file__).resolve().parent; package=sources.parent
    files=list(sources.glob('*.py'))+[sources/'_excess.cpp']+list((root/'models').glob('*.npz'))
    hashes={str(p.resolve().relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    content=dict(version='0.4.0', created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                 sha256=hashes, main_starts=MAIN_STARTS, larger_starts=LARGE_STARTS,
                 extended_search_first_replicates=3, old_interval_comparison_first_replicates=8,
                 known_participation_first_replicates=8,
                 decision_gates=dict(simulation_failures=0,
                   fixed_parameter_acceptance='Report every case with Wilson uncertainty; flag rates below 90%. No uniform claim follows from a finite case collection.',
                   response_coverage='Report every case, empty sets, and independent-oracle uncertainty; flag rates below 95%.',
                   selected_endpoint_accuracy='Audit both panels and both banks; flag every error exceeding the response allowance.',
                   search='Flag endpoint movements greater than 0.005 percentage points under increased search.',
                   information='Assess interval width jointly with coverage; report paired linkage differences without requiring a positive result.'),
                 original_confirmation_seeds_used=False)
    dest=root/'source_and_model_freeze.json'
    if dest.exists():
        prior=json.loads(dest.read_text())
        if prior['sha256'] != hashes:
            raise RuntimeError('Source/model freeze changed')
        return prior
    dest.write_text(json.dumps(content, indent=2)+'\n'); return content

def verify_freeze(root):
    frozen=json.loads((Path(root)/'source_and_model_freeze.json').read_text())
    package=Path(__file__).resolve().parent.parent
    for path, digest in frozen['sha256'].items():
        if hashlib.sha256((package/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen input changed: {path}')

def evaluate_job(job):
    root=Path(job['root']); bank=job['bank']; panel=job['panel']; case=job['case']; tid=case['truth_id']
    model=MODELS[bank, panel]; old=OLD[bank, panel]; start=time.perf_counter(); xtrue=encode(case['parameters']); rows=[]
    with np.load(root/'evaluation'/f'truth_{tid:02d}.npz', allow_pickle=False) as z:
        data=z[panel]; seeds=z['seeds']
    for rep, y in enumerate(data):
        budgets=(4, MAIN_STARTS[panel], LARGE_STARTS[panel]) if rep < 3 else (4, MAIN_STARTS[panel])
        q=model.quantities(xtrue, y)
        oldscore=float(np.sum((old_basis(xtrue)@old.wcoef-y@old.white)**2))
        common=dict(bank=bank, panel=panel, truth_id=tid, name=case['name'], kind=case['kind'],
                    status=case['status'], replicate=rep, seed=int(seeds[rep]))
        for fit in model.fit(y, budgets=budgets):
            rows.append(dict(**common, method='v0.4', participation='unknown',
                             true_parameter_accepted=q['ratio'] <= 1, true_score=q['score'], true_cutoff=q['cutoff'],
                             old_true_parameter_accepted=oldscore <= old.cutoff, **fit))
        if rep < 8:
            oldfit=old.fit(y, budgets=(8,))[0]
            rows.append(dict(**common, method='v0.3', participation='unknown',
                             true_parameter_accepted=oldscore <= old.cutoff, true_score=oldscore,
                             true_cutoff=old.cutoff, **oldfit))
        if panel == 'O6' and rep < 8:
            fit=model.fit(y, budgets=(MAIN_STARTS[panel],), fixed_selection=float(xtrue[-1]))[0]
            rows.append(dict(**common, method='v0.4', participation='known_diagnostic',
                             true_parameter_accepted=q['ratio'] <= 1, true_score=q['score'], true_cutoff=q['cutoff'], **fit))
    path=root/'fits'/f'bank_{bank}_{panel}_truth_{tid:02d}.json'
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n'); temp.replace(path)
    return dict(bank=bank, panel=panel, truth_id=tid, rows=len(rows), seconds=time.perf_counter()-start)

def run(root, previous, workers=8):
    root=Path(root); verify_freeze(root); (root/'fits').mkdir(exist_ok=True)
    pending=[]
    for bank in (0, 1):
        for panel in PANELS:
            for case in cases():
                if not (root/'fits'/f'bank_{bank}_{panel}_truth_{case["truth_id"]:02d}.json').exists():
                    pending.append(dict(root=str(root), bank=bank, panel=panel, case=case))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init, initargs=(str(root), str(previous))) as pool:
        futures=[pool.submit(evaluate_job, job) for job in pending]
        for i, future in enumerate(as_completed(futures)):
            result=future.result()
            progress=dict(done=i+1, total=len(futures), seconds=time.perf_counter()-start, last=result)
            print(json.dumps(progress), flush=True)
            (root/'progress_fitting.json').write_text(json.dumps(progress, indent=2)+'\n')
    summarize(root)

def paired(rows, oracle):
    linked={(r['truth_id'], r['replicate']):r for r in rows if r['panel']=='O6'}
    unlinked={(r['truth_id'], r['replicate']):r for r in rows if r['panel']=='O7'}
    widths=[]; covers=[]
    for key, a in linked.items():
        b=unlinked[key]; mean=oracle[a['truth_id']]['mean']
        ca=not a['empty'] and a['lower'] <= mean <= a['upper']
        cb=not b['empty'] and b['lower'] <= mean <= b['upper']
        covers.append(int(ca)-int(cb))
        if not a['empty'] and not b['empty']:
            widths.append(b['width']-a['width'])
    return dict(n_pairs=len(linked), joint_nonempty=len(widths),
                O7_minus_O6_mean_width_pp=float(np.mean(widths)),
                paired_width_mcse_pp=float(np.std(widths, ddof=1)/np.sqrt(len(widths))),
                O6_minus_O7_coverage_pp=float(100*np.mean(covers)))

def summarize(root):
    root=Path(root); dest=root/'summary'; dest.mkdir(exist_ok=True); rows=[]
    for path in sorted((root/'fits').glob('*.json')):
        rows.extend(json.loads(path.read_text()))
    oracle={}
    for case in cases():
        tid=case['truth_id']
        with np.load(root/'oracle'/f'truth_{tid:02d}.npz', allow_pickle=False) as z:
            sample=z['response']
        se=float(sample.std(ddof=1)/np.sqrt(len(sample)))
        oracle[tid]=dict(truth_id=tid, name=case['name'], status=case['status'], n=len(sample),
                         mean=float(sample.mean()), mcse=se, half_99=float(t.ppf(.995, len(sample)-1)*se))
    write_csv(dest/'oracle.csv', list(oracle.values()))
    main=[r for r in rows if r['method']=='v0.4' and r['participation']=='unknown' and r['starts']==MAIN_STARTS[r['panel']]]
    bycase=[]; byscope=[]; links=[]; search=[]; versions=[]; participation=[]; bankdiff=[]
    for bank in (0, 1):
        for panel in PANELS:
            subset=[r for r in main if r['bank']==bank and r['panel']==panel]
            for case in cases():
                cell=[r for r in subset if r['truth_id']==case['truth_id']]
                if not cell:
                    raise RuntimeError('Incomplete fitted results')
                s=stats(cell, oracle)
                accepted=sum(r['true_parameter_accepted'] for r in cell); low, high=wilson(accepted, len(cell))
                s.update(parameter_acceptance_low_pct=100*low, parameter_acceptance_high_pct=100*high,
                         old_parameter_acceptance_pct=float(100*np.mean([r['old_true_parameter_accepted'] for r in cell])),
                         cutoff=cell[0]['true_cutoff'])
                bycase.append(dict(bank=bank, panel=panel, truth_id=case['truth_id'], name=case['name'],
                                   status=case['status'], kind=case['kind'], **s))
            for label in ['all', 'previously_seen_setting', 'new_setting']:
                group=subset if label=='all' else [r for r in subset if r['status']==label]
                byscope.append(dict(bank=bank, panel=panel, scope=label, **stats(group, oracle)))
            ix={(r['truth_id'], r['replicate'], r['starts']):r for r in rows
                if r['bank']==bank and r['panel']==panel and r['method']=='v0.4' and r['participation']=='unknown'}
            for small, large in [(4, MAIN_STARTS[panel]), (MAIN_STARTS[panel], LARGE_STARTS[panel])]:
                moves=[]; resolved=0; shrinking=0; pairs=0
                for (tid, rep, budget), big in ix.items():
                    if budget != large:
                        continue
                    little=ix[tid, rep, small]; pairs+=1
                    resolved+=int(little['empty'] and not big['empty'])
                    if not little['empty'] and not big['empty']:
                        moves.append(max(little['lower']-big['lower'], big['upper']-little['upper']))
                        shrinking+=int(big['lower']>little['lower']+1e-9 or big['upper']<little['upper']-1e-9)
                search.append(dict(bank=bank, panel=panel, small=small, large=large, n=pairs, new_nonempty=resolved,
                                   shrinking=shrinking, max_endpoint_movement_pp=float(np.max(moves)),
                                   p95_endpoint_movement_pp=float(np.quantile(moves, .95)),
                                   n_above_0_005_pp=int(np.sum(np.asarray(moves)>.005))))
            old=[r for r in rows if r['bank']==bank and r['panel']==panel and r['method']=='v0.3']
            new=[r for r in subset if r['replicate']<8]
            for method, selected in [('v0.3', old), ('v0.4', new)]:
                versions.append(dict(bank=bank, panel=panel, method=method, **stats(selected, oracle)))
        for label in ['all', 'previously_seen_setting', 'new_setting']:
            subset=[r for r in main if r['bank']==bank and (label=='all' or r['status']==label)]
            links.append(dict(bank=bank, scope=label, **paired(subset, oracle)))
        known=[r for r in rows if r['bank']==bank and r['method']=='v0.4' and r['participation']=='known_diagnostic']
        index={(r['truth_id'], r['replicate']):r for r in main if r['bank']==bank and r['panel']=='O6'}
        for r in known:
            unknown=index[r['truth_id'], r['replicate']]
            participation.append(dict(bank=bank, truth_id=r['truth_id'], replicate=r['replicate'],
                                       known_empty=r['empty'], unknown_empty=unknown['empty'],
                                       width_cost=None if r['empty'] or unknown['empty'] else unknown['width']-r['width']))
    for panel in PANELS:
        a={(r['truth_id'], r['replicate']):r for r in main if r['bank']==0 and r['panel']==panel}
        b={(r['truth_id'], r['replicate']):r for r in main if r['bank']==1 and r['panel']==panel}
        diffs=[max(abs(r['lower']-b[k]['lower']), abs(r['upper']-b[k]['upper'])) for k, r in a.items()
               if not r['empty'] and not b[k]['empty']]
        bankdiff.append(dict(panel=panel, pairs=len(a), nonempty_pairs=len(diffs),
                             mean_endpoint_difference_pp=float(np.mean(diffs)), p95_endpoint_difference_pp=float(np.quantile(diffs, .95))))
    for name, data in [('by_case', bycase), ('by_scope', byscope), ('linked_vs_unlinked', links),
                        ('search_stability', search), ('paired_version_comparison', versions),
                        ('participation_cost', participation), ('independent_banks', bankdiff)]:
        write_csv(dest/f'{name}.csv', data)
    failures=[dict(bank=r['bank'], panel=r['panel'], name=r['name'], parameter_acceptance_pct=r['true_parameter_acceptance_pct'],
                   response_coverage_pct=r['coverage_pct'], empty_pct=r['empty_pct'])
              for r in bycase if r['true_parameter_acceptance_pct']<90 or r['coverage_pct']<95]
    summary=dict(rows=len(rows), main_intervals=len(main), validation_markets=1920,
                 fitted_banks=2, new_settings=12, reused_development_settings=12,
                 cases_flagged_for_low_retention_or_coverage=failures,
                 banks_share_validation_markets=True, original_confirmation_seeds_used=False)
    (dest/'summary_counts.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--previous', default='results/expanded_pilot'); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--freeze', action='store_true'); p.add_argument('--summary-only', action='store_true')
    a=p.parse_args()
    if a.freeze:
        print(json.dumps(freeze(a.root)))
    elif a.summary_only:
        summarize(a.root)
    else:
        run(a.root, a.previous, a.workers)
