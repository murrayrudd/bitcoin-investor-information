"""Prespecified summaries for the frozen v0.6.0 confirmation experiment.

Bootstrap units are parameter locations. The same stratified draws preserve
every market, panel, bank, and reference comparison within each location.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import confirmation_runner as run
from pilot.repair_fit import RepairedFitter, basis


def load_markets(role, cases):
    n = run.N_EVAL if role == 'evaluation' else run.N_REF
    response = np.empty((len(cases), n))
    features = {p: [] for p in run.PANELS}
    diagnostics = []
    for c in cases:
        i = c['case_id']
        parts = sorted((run.ROOT/role).glob(f'case_{i:03d}_*.npz'))
        sample, seeds = [], []
        ys = {p: [] for p in run.PANELS}
        for path in parts:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
                if meta['failures'] or meta['attempted'] != meta['successful']:
                    raise RuntimeError('Failed simulation checkpoint ' + str(path))
                diagnostics.append(meta)
                sample.append(z['response']); seeds.append(z['seeds'])
                if role == 'evaluation':
                    for p in run.PANELS: ys[p].append(z[p])
        expected = np.array([run.seed(role, i, r) for r in range(n)])
        np.testing.assert_array_equal(np.concatenate(seeds), expected)
        response[i] = np.concatenate(sample)
        if role == 'evaluation':
            for p in run.PANELS: features[p].append(np.concatenate(ys[p]))
    return response, {p: np.array(v) for p, v in features.items()}, diagnostics


class Bootstrap:
    def __init__(self, cases, n=2000):
        self.n = n; self.weights = {}; self.draws = {}
        rng = np.random.default_rng(run.seed('bootstrap', 0))
        for name in run.COMPONENTS:
            ids = np.array([c['case_id'] for c in cases if c['component'] == name])
            indices = rng.integers(0, len(ids), (n, len(ids)))
            counts = np.array([np.bincount(row, minlength=len(ids)) for row in indices])/len(ids)
            w = np.zeros(len(cases)); w[ids] = 1/len(ids)
            b = np.zeros((n, len(cases))); b[:, ids] = counts
            self.weights[name] = w; self.draws[name] = b
        weights = [.75, 1/12, 1/12, 1/12]
        self.weights['primary'] = sum(w*self.weights[g] for w, g in zip(weights, run.COMPONENTS))
        self.draws['primary'] = sum(w*self.draws[g] for w, g in zip(weights, run.COMPONENTS))

    def estimate(self, group, numerator, denominator=None, scale=1.):
        w, b = self.weights[group], self.draws[group]
        point = w@numerator; values = b@numerator
        if denominator is not None:
            den, dens = w@denominator, b@denominator
            point = point/den if den > 0 else np.nan
            values = np.divide(values, dens, out=np.full_like(values, np.nan), where=dens > 0)
        point *= scale; values *= scale
        valid = values[np.isfinite(values)]
        return dict(estimate=float(point) if np.isfinite(point) else None,
                    mcse=float(valid.std(ddof=1)) if len(valid) > 1 else None,
                    ci95_low=float(np.quantile(valid, .025)) if len(valid) else None,
                    ci95_high=float(np.quantile(valid, .975)) if len(valid) else None,
                    bootstrap_valid=len(valid))


def summarize():
    run.verify()
    for stage in ('generate', 'fit', 'audit'):
        if not (run.ROOT/f'completed_{stage}.json').exists(): raise RuntimeError('Incomplete ' + stage)
    out = run.ROOT/'summary'; out.mkdir(exist_ok=True)
    spec = run.read_json(run.ROOT/'specification.json'); cases = spec['cases']
    evaluation, features, diagnostics = load_markets('evaluation', cases)
    reference, _, ref_diagnostics = load_markets('reference', cases)
    diagnostics += ref_diagnostics
    means = reference.mean(axis=1); se = reference.std(axis=1, ddof=1)/np.sqrt(run.N_REF)
    half = t.ppf(.995, run.N_REF-1)*se
    refs = pd.DataFrame([dict(case_id=c['case_id'], component=c['component'],
        mean_response_pp=means[i], reference_se_pp=se[i], ci99_low=means[i]-half[i],
        ci99_high=means[i]+half[i], n_reference=run.N_REF,
        evaluation_mean_response_pp=evaluation[i].mean()) for i, c in enumerate(cases)])
    refs.to_csv(out/'reference_means.csv', index=False)
    shape = (2, len(run.PANELS), len(cases), run.N_EVAL)
    lower = np.full(shape, np.nan); upper = np.full(shape, np.nan)
    empty = np.ones(shape, bool); seen = np.zeros(shape, bool)
    primary_records, check_records = [], []
    endpoint_ratio_max = 0.; optimizer_failures = 0; optimizations = 0
    models = {(b,p): RepairedFitter(run.ROOT/'models'/f'bank_{b}_{p}.npz')
              for b in (0,1) for p in run.PANELS}
    for path in sorted((run.ROOT/'fits').glob('*.json')):
        rows = run.read_json(path)
        main = {r['replicate']: r for r in rows if r['method'] == 'primary'}
        for r in rows:
            b, p, i, rep = r['bank'], r['panel'], r['case_id'], r['replicate']
            j = run.PANELS.index(p); ix = (b,j,i,rep)
            if r['method'] == 'primary':
                if seen[ix]: raise RuntimeError('Duplicate primary fit')
                seen[ix] = True; empty[ix] = r['empty']
                if not r['empty']:
                    lower[ix], upper[ix] = r['lower'], r['upper']
                    if upper[ix] < lower[ix]: raise RuntimeError('Negative interval width')
                    m = models[b,p]; y = features[p][i,rep]
                    for side in ('lower','upper'):
                        v = basis(np.asarray(r[side+'_x']))
                        resid = v@m.wcoef-y@m.white
                        ratio = float(resid@resid/np.exp(v@m.cutoff_coef))
                        endpoint_ratio_max = max(endpoint_ratio_max, ratio)
                        if ratio > 1+1.01e-7: raise RuntimeError('Infeasible saved endpoint')
                optimizer_failures += r['counts']['optimizer_failures']
                optimizations += r['counts']['endpoint_optimizations']+r['counts']['score_minimizations']
                primary_records.append(dict(bank=b,panel=p,case_id=i,component=cases[i]['component'],
                    replicate=rep,empty=r['empty'],lower_pp=r['lower'],upper_pp=r['upper'],
                    width_pp=r['width'],best_ratio=r['best_ratio'],response_allowance_pp=r['response_error_pp']))
            else:
                a = main[rep]; both = not a['empty'] and not r['empty']
                change = max(abs(r['lower']-a['lower']),abs(r['upper']-a['upper'])) if both else None
                recovered = bool(a['empty'] and not r['empty'])
                check_records.append(dict(bank=b,panel=p,case_id=i,replicate=rep,method=r['method'],
                    primary_empty=a['empty'],check_empty=r['empty'],empty_recovered=recovered,
                    lower_change_pp=r['lower']-a['lower'] if both else None,
                    upper_change_pp=r['upper']-a['upper'] if both else None,
                    max_endpoint_change_pp=change,
                    exceeds_tolerance=recovered or (change is not None and change > .005)))
    if not seen.all() or int(seen.sum()) != 131072: raise RuntimeError('Incomplete primary fits')
    widths = upper-lower
    covered = (~empty)&(lower<=means[None,None,:,None])&(upper>=means[None,None,:,None])
    definite = (~empty)&(lower<=(means-half)[None,None,:,None])&(upper>=(means+half)[None,None,:,None])
    possible = (~empty)&(lower<=(means+half)[None,None,:,None])&(upper>=(means-half)[None,None,:,None])
    retained = np.zeros(shape, bool)
    for b in (0,1):
        for j,p in enumerate(run.PANELS):
            m = models[b,p]
            for i,c in enumerate(cases):
                v = basis(np.asarray(c['x'])); residual = features[p][i]@m.white-v@m.wcoef
                retained[b,j,i] = np.sum(residual**2,axis=1)/np.exp(v@m.cutoff_coef)<=1
    for row in primary_records:
        ix = (row['bank'],run.PANELS.index(row['panel']),row['case_id'],row['replicate'])
        row.update(coverage=bool(covered[ix]),definite_coverage=bool(definite[ix]),
                   possible_coverage=bool(possible[ix]),true_parameter_retained=bool(retained[ix]))
    pd.DataFrame(primary_records).to_csv(out/'primary_intervals.csv',index=False)
    checks = pd.DataFrame(check_records); checks.to_csv(out/'search_checks.csv',index=False)
    bootstrap = Bootstrap(cases)
    groups = ('primary',)+run.COMPONENTS
    panel_records, location_records = [], []
    for b in (0,1):
        for j,p in enumerate(run.PANELS):
            numerator = dict(coverage=covered[b,j].mean(axis=1),
                definite_coverage=definite[b,j].mean(axis=1),possible_coverage=possible[b,j].mean(axis=1),
                true_parameter_retention=retained[b,j].mean(axis=1),empty_rate=empty[b,j].mean(axis=1),
                mean_width_pp=np.nan_to_num(widths[b,j],nan=0.).sum(axis=1)/run.N_EVAL)
            nonempty = (~empty[b,j]).mean(axis=1)
            for i,c in enumerate(cases):
                location_records.append(dict(bank=b,panel=p,case_id=i,component=c['component'],
                    n_nonempty=int((~empty[b,j,i]).sum()),
                    **{('width_sum_divided_by_64_pp' if k=='mean_width_pp' else k):float(v[i]) for k,v in numerator.items()},
                    conditional_mean_width_pp=float(np.nanmean(widths[b,j,i])) if nonempty[i] else None))
            for group in groups:
                ids = np.flatnonzero(bootstrap.weights[group])
                for metric, num in numerator.items():
                    est = bootstrap.estimate(group,num,nonempty if metric=='mean_width_pp' else None)
                    panel_records.append(dict(group=group,bank=b,panel=p,metric=metric,
                        n_settings=len(ids),n_markets=len(ids)*run.N_EVAL,
                        n_nonempty=int((~empty[b,j,ids]).sum()),**est))
                vals = widths[b,j,ids]; vals = vals[np.isfinite(vals)]
                panel_records.append(dict(group=group,bank=b,panel=p,metric='median_width_pp',
                    n_settings=len(ids),n_markets=len(ids)*run.N_EVAL,n_nonempty=len(vals),
                    estimate=float(np.median(vals)) if len(vals) else None,
                    mcse=None,ci95_low=None,ci95_high=None,bootstrap_valid=None))
    panels = pd.DataFrame(panel_records); panels.to_csv(out/'panel_summary.csv',index=False)
    pd.DataFrame(location_records).to_csv(out/'setting_summary.csv',index=False)
    contrast_records, contrast_locations = [], []
    for b in (0,1):
        for base,expanded in spec['contrasts']:
            a,j = run.PANELS.index(base),run.PANELS.index(expanded)
            joint = (~empty[b,a])&(~empty[b,j])
            fraction = joint.mean(axis=1)
            diff = np.where(joint,widths[b,a]-widths[b,j],0.).mean(axis=1)
            baseline = np.where(joint,widths[b,a],0.).mean(axis=1)
            for i,c in enumerate(cases):
                contrast_locations.append(dict(bank=b,comparison=base,expanded=expanded,case_id=i,
                    component=c['component'],joint_nonempty_fraction=float(fraction[i]),
                    joint_difference_sum_divided_by_64_pp=float(diff[i]),
                    joint_baseline_width_sum_divided_by_64_pp=float(baseline[i])))
            for group in groups:
                for metric,num,den,scale in [('joint_nonempty_fraction',fraction,None,1.),
                    ('width_reduction_pp',diff,fraction,1.),('width_reduction_percent',diff,baseline,100.)]:
                    contrast_records.append(dict(group=group,bank=b,comparison=base,expanded=expanded,
                        metric=metric,**bootstrap.estimate(group,num,den,scale)))
    contrasts = pd.DataFrame(contrast_records); contrasts.to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(contrast_locations).to_csv(out/'setting_contrasts.csv',index=False)
    audit_records = []
    for e in run.read_json(run.ROOT/'audit/selected_endpoints.json'):
        row = dict(e); row.pop('x',None)
        if not e['missing']:
            with np.load(run.ROOT/'audit'/f"endpoint_{e['index']:03d}.npz",allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'])); diagnostics.append(meta)
                if meta['failures'] or len(z['response']) != run.N_AUDIT: raise RuntimeError('Incomplete audit')
                sample = z['response']; mean = float(sample.mean()); err = abs(mean-e['predicted_response'])
                refse = float(sample.std(ddof=1)/np.sqrt(len(sample)))
                refhalf = float(t.ppf(.995,len(sample)-1)*refse)
                m = models[e['bank'],e['panel']]; v = basis(np.array(e['x']))
                residual = z[e['panel']]@m.white-v@m.wcoef
                acceptance = np.sum(residual**2,axis=1)/np.exp(v@m.cutoff_coef)<=1
                row.update(n_markets=len(sample),reference_mean_pp=mean,reference_se_pp=refse,
                    ci99_low=mean-refhalf,ci99_high=mean+refhalf,absolute_error_pp=err,
                    point_within_allowance=err<=e['allowance'],
                    full_ci99_within_allowance=err+refhalf<=e['allowance'],
                    ci99_overlaps_allowance=max(0.,err-refhalf)<=e['allowance'],
                    heldout_parameter_retention=float(acceptance.mean()))
        audit_records.append(row)
    audits = pd.DataFrame(audit_records); audits.to_csv(out/'endpoint_audits.csv',index=False)
    integrity = dict(status='completed',summary_utc=run.now(),primary_intervals=int(seen.sum()),
        evaluation_markets=int(evaluation.size),reference_markets=int(reference.size),
        audit_markets=int(audits['n_markets'].sum()),missing_audit_endpoints=int(audits['missing'].sum()),
        max_clearing=max(d['max_clearing'] for d in diagnostics),max_cash=max(d['max_cash'] for d in diagnostics),
        max_kkt=max(d['max_kkt'] for d in diagnostics),multiple_root_periods=sum(d['multiple_root_periods'] for d in diagnostics),
        max_primary_endpoint_ratio=endpoint_ratio_max,primary_optimizer_calls=optimizations,
        primary_optimizer_non_success=optimizer_failures,search_checks=len(checks),
        search_tolerance_exceedances=int(checks.exceeds_tolerance.sum()),
        empty_intervals_recovered_by_checks=int(checks.empty_recovered.sum()),
        endpoint_point_allowance_failures=int((audits.point_within_allowance==False).sum()),
        endpoint_full_ci99_allowance_failures=int((audits.full_ci99_within_allowance==False).sum()),
        bootstrap=dict(resamples=2000,unit='parameter setting',stratified=True,seed=run.seed('bootstrap',0),
            same_indices_for_all_contrasts=True,boundary_caveat='A degenerate bootstrap interval at zero or one reflects no observed setting variation; it does not prove a boundary probability.'),
        units='Response and interval widths: percentage points. Rates: fractions. Relative width reduction: percent.',
        inference_scope=spec['inference_claim'],source_freeze_verified=True)
    run.atomic_json(out/'integrity.json',integrity)
    run.verify()
    print(json.dumps(integrity,indent=2))


if __name__ == '__main__': summarize()
