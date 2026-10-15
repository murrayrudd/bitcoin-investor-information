"""Smooth parameter-dependent test inversion with independent calibration.

The rank correction protects each declared random design component, not every
fixed parameter. Continuous endpoint search remains a numerical approximation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc, t
from sklearn.preprocessing import PolynomialFeatures
from .observations import PANELS
from .inference_features import vector
from .expanded_fit import rank_threshold
from .repair_design import ALPHA_OBS, ALPHA_RESPONSE, ALPHA_REFERENCE, GROUPS, COUNTS

POLY=PolynomialFeatures(4).fit(np.zeros((1, 9)))
POWERS=POLY.powers_
DEGREES=POWERS.sum(axis=1)
INDICES=np.full((len(POWERS), 4), 9, dtype=int)
for j, p in enumerate(POWERS):
    ids=np.repeat(np.arange(9), p); INDICES[j, :len(ids)]=ids
LOOKUP={tuple(p):i for i, p in enumerate(POWERS)}
LOWER=np.zeros((9, len(POWERS)), dtype=int)
for d in range(9):
    for j, p in enumerate(POWERS):
        if p[d]:
            lowered=p.copy(); lowered[d]-=1; LOWER[d, j]=LOOKUP[tuple(lowered)]

def basis(x, derivative=False):
    xx=np.r_[x, 1.]; v=np.prod(xx[INDICES], axis=1)
    if not derivative:
        return v
    return v, v[LOWER]*POWERS.T

def load_settings(folder):
    rows=[]; names=None
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['failures'] or meta['attempted'] != meta['successful']:
                raise RuntimeError(f'Incomplete simulation file {path}')
            row={k:z[k] for k in z.files if k != 'metadata'}
            row['path']=str(path); rows.append(row)
            if meta['feature_names']:
                names=meta['feature_names']
    if not rows:
        raise RuntimeError(f'Empty simulation bank {folder}')
    return rows, names

def regression(x, y, weights, degree, participation=True):
    mask=DEGREES <= degree
    if not participation:
        mask &= POWERS[:, 8] == 0
    a=POLY.transform(x)[:, mask]; w=np.asarray(weights, float)
    w=w/w.mean(); penalty=.1*np.eye(a.shape[1]); penalty[0, 0]=0.
    fitted=np.linalg.solve(a.T@(w[:, None]*a)+penalty, a.T@(w.reshape((-1,)+(1,)*(y.ndim-1))*y))
    coef=np.zeros((len(POWERS),)+y.shape[1:]); coef[mask]=fitted
    return coef

def covariance(rows, panel):
    residuals=[r[panel]-r[panel].mean(axis=0) for r in rows]
    cov=sum(v.T@v for v in residuals)/sum(len(v)-1 for v in residuals)
    cov=.9*cov+.1*np.diag(np.diag(cov))
    return cov

def fit_log_quantile(x, log_scores, tau=1-ALPHA_OBS):
    # Each setting contributes equally and each independent market one row.
    mask=DEGREES <= 2
    a=POLY.transform(x)[:, mask]
    targets=np.asarray(log_scores)
    if targets.ndim != 2:
        raise ValueError('Quantile bank must have equal replications per setting')
    initial=np.zeros(a.shape[1]); initial[0]=np.quantile(targets, tau)
    eps=.03; penalty=.002
    def objective(coef):
        prediction=a@coef; residual=targets-prediction[:, None]
        value=np.mean(tau*residual+eps*np.logaddexp(0., -residual/eps))
        gradient=a.T@np.mean(expit(-residual/eps)-tau, axis=1)/len(a)
        value+=penalty*np.sum(coef[1:]**2)
        gradient[1:]+=2*penalty*coef[1:]
        return value, gradient
    fit=minimize(objective, initial, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=1200, ftol=1e-12, gtol=1e-7))
    if not fit.success:
        raise RuntimeError(f'Conditional quantile fit failed: {fit.message}')
    result=np.zeros(len(POWERS)); result[mask]=fit.x
    return result, dict(success=bool(fit.success), iterations=int(fit.nit), objective=float(fit.fun))

def train_bank(root, previous, bank):
    root=Path(root); previous=Path(previous); folder=root/f'bank_{bank}'
    # Historical validation and all historical diagnostics are development
    # information. Old train/check/response files improve precision only.
    old, names=load_settings(previous/f'bank_{bank}'/'train')
    oldcheck, _=load_settings(previous/f'bank_{bank}'/'check')
    extra, _=load_settings(folder/'train'); check, _=load_settings(folder/'check')
    train=old+oldcheck+extra
    quantile, _=load_settings(folder/'quantile')
    calibration, _=load_settings(folder/'calibration')
    response_ref, _=load_settings(folder/'response')
    expected={'extra':812, 'check':COUNTS['check'], 'quantile':COUNTS['quantile'],
              'calibration':3*COUNTS['calibration_per_group'], 'response_ref':3*COUNTS['response_per_group']}
    for label, rows in [('extra', extra), ('check', check), ('quantile', quantile),
                        ('calibration', calibration), ('response_ref', response_ref)]:
        if len(rows) != expected[label]:
            raise RuntimeError(f'Incomplete {label} bank: {len(rows)} != {expected[label]}')
    x=np.array([r['x'] for r in train]); weights=np.array([len(r['response']) for r in train])
    xc=np.array([r['x'] for r in check]); fc=POLY.transform(xc)
    covs={p:covariance(train, p) for p in PANELS}
    candidates={}; degree_losses={}
    for degree in (3, 4):
        candidates[degree]={}
        loss=[]
        for panel in PANELS:
            mean=np.array([r[panel].mean(axis=0) for r in train])
            coef=regression(x, mean, weights, degree)
            for j, name in enumerate(names[panel]):
                if name.startswith('signal_'):
                    coef[:, j]=0.
            candidates[degree][panel]=coef
            checkmean=np.array([r[panel].mean(axis=0) for r in check])
            standardized=(checkmean-fc@coef)/np.sqrt(np.diag(covs[panel]))
            loss.append(np.mean(standardized**2))
        degree_losses[str(degree)]=float(np.mean(loss))
    degree=min((3, 4), key=lambda d:degree_losses[str(d)])

    # The population response cannot depend on observation participation.
    oldresponse, _=load_settings(previous/f'bank_{bank}'/'response')
    rt=train+oldresponse
    xr=np.array([r['x'] for r in rt]); yr=np.array([r['response'].mean() for r in rt])
    wr=np.array([len(r['response']) for r in rt])
    response_candidates={d:regression(xr, yr, wr, d, participation=False) for d in (3, 4)}
    checkresponse=np.array([r['response'].mean() for r in check])
    response_losses={str(d):float(np.mean((fc@c-checkresponse)**2)) for d, c in response_candidates.items()}
    response_degree=min((3, 4), key=lambda d:response_losses[str(d)])
    response_coef=response_candidates[response_degree]
    errors={g:[] for g in GROUPS}
    for r in response_ref:
        sample=r['response']; se=sample.std(ddof=1)/np.sqrt(len(sample))
        half=t.ppf(1-ALPHA_REFERENCE/2, len(sample)-1)*se
        err=abs(sample.mean()-basis(r['x'])@response_coef)+half
        errors[GROUPS[int(r['stratum'])]].append(float(err))
    allowances={g:rank_threshold(values, ALPHA_RESPONSE) for g, values in errors.items()}
    response_error=max(allowances.values())
    diagnostics=dict(bank=bank, mean_degree=degree, mean_degree_check_mse_sd=degree_losses,
                     response_degree=response_degree, response_degree_check_mse_pp=response_losses,
                     response_error_by_group_pp=allowances, response_error_pp=response_error,
                     new_train_settings=len(extra), total_train_settings=len(train), panels={})
    dest=root/'models'; dest.mkdir(exist_ok=True)
    xq=np.array([r['x'] for r in quantile]); fq=POLY.transform(xq)
    xcal=np.array([r['x'] for r in calibration]); fcal=POLY.transform(xcal)
    for panel in PANELS:
        coef=candidates[degree][panel]; white=np.linalg.cholesky(np.linalg.inv(covs[panel]))
        qdata=np.array([r[panel] for r in quantile])
        scores=np.sum(((qdata-(fq@coef)[:, None, :])@white)**2, axis=2)
        cutoff_coef, fit_diagnostic=fit_log_quantile(xq, np.log(np.maximum(scores, 1e-12)))
        caldata=np.array([r[panel][0] for r in calibration])
        calscores=np.sum(((caldata-fcal@coef)@white)**2, axis=1)
        ratios=calscores/np.exp(fcal@cutoff_coef)
        correction={g:rank_threshold(ratios[[int(r['stratum']) == i for r in calibration]], ALPHA_OBS)
                    for i, g in enumerate(GROUPS)}
        factor=max(correction.values())
        cutoff_coef[0]+=np.log(factor)
        diagnostic=dict(quantile_fit=fit_diagnostic, rank_correction_by_group=correction,
                        applied_rank_correction=factor,
                        calibration_cutoff_min=float(np.exp(fcal@cutoff_coef).min()),
                        calibration_cutoff_max=float(np.exp(fcal@cutoff_coef).max()))
        diagnostics['panels'][panel]=diagnostic
        meta=dict(version='0.4.0', bank=bank, panel=panel, feature_names=names[panel],
                  mean_degree=degree, response_degree=response_degree, response_error_pp=response_error,
                  response_error_by_group_pp=allowances, calibration=diagnostic,
                  coverage_scope='Component-marginal rank calibration; parameter-dependent cutoff; fixed-parameter performance requires independent checks; no uniform continuous-domain guarantee.')
        np.savez_compressed(dest/f'bank_{bank}_{panel}.npz', coef=coef, white=white,
                            response_coef=response_coef, cutoff_coef=cutoff_coef,
                            train_x=x, metadata=np.array(json.dumps(meta)))
    (dest/f'bank_{bank}_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    return diagnostics

class RepairedFitter:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.coef=z['coef']; self.white=z['white']; self.response_coef=z['response_coef']
            self.cutoff_coef=z['cutoff_coef']; self.train_x=z['train_x']
            self.meta=json.loads(str(z['metadata']))
        self.error=self.meta['response_error_pp']; self.wcoef=self.coef@self.white
        sobol=qmc.Sobol(9, scramble=True, seed=994001).random_base2(10)*2-1
        self.pool=np.vstack([sobol, self.train_x])
        features=POLY.transform(self.pool)
        self.pool_mean=features@self.wcoef
        self.pool_cutoff=np.exp(features@self.cutoff_coef)

    def quantities(self, x, y, derivative=False):
        if derivative:
            v, j=basis(x, True)
        else:
            v=basis(x)
        residual=v@self.wcoef-np.asarray(y)@self.white
        cutoff=float(np.exp(v@self.cutoff_coef)); score=float(residual@residual)
        out=dict(score=score, cutoff=cutoff, ratio=score/cutoff,
                 response=float(v@self.response_coef))
        if derivative:
            out['ratio_grad']=(2*(j@self.wcoef)@residual-score*(j@self.cutoff_coef))/cutoff
            out['response_grad']=j@self.response_coef
        return out

    def fit(self, y, budgets=(4, 12), fixed_selection=None):
        y=np.asarray(y, float)
        if y.shape != (self.coef.shape[1],) or not np.isfinite(y).all():
            raise ValueError('Invalid released summary')
        if sorted(set(budgets)) != list(budgets) or min(budgets) < 1:
            raise ValueError('Search budgets must be increasing positive integers')
        bounds=[(-1., 1.)]*9; pool=self.pool
        if fixed_selection is None:
            poolmean=self.pool_mean; poolcutoff=self.pool_cutoff
        else:
            if not -1 <= fixed_selection <= 1:
                raise ValueError('Participation outside domain')
            bounds[-1]=(fixed_selection, fixed_selection)
            pool=pool.copy(); pool[:, -1]=fixed_selection; f=POLY.transform(pool)
            poolmean=f@self.wcoef; poolcutoff=np.exp(f@self.cutoff_coef)
        ratios=np.sum((poolmean-y@self.white)**2, axis=1)/poolcutoff
        order=np.argsort(ratios); chosen=[]; starts=[]
        for i in range(max(budgets)):
            if i % 3 == 0:
                idx=next(int(j) for j in order if int(j) not in chosen)
            else:
                candidates=order[:512]
                distances=np.min(np.sum((pool[candidates, None, :]-pool[chosen][None, :, :])**2, axis=2), axis=1)
                idx=int(candidates[np.argmax(distances)])
            chosen.append(idx); starts.append(pool[idx])
        cache={}; yw=y@self.white
        def evaluate(x):
            if 'x' not in cache or not np.array_equal(x, cache['x']):
                v, j=basis(x, True); residual=v@self.wcoef-yw
                cutoff=np.exp(v@self.cutoff_coef); score=residual@residual
                cache.update(x=x.copy(), ratio=float(score/cutoff), score=float(score), cutoff=float(cutoff),
                             ratio_grad=(2*(j@self.wcoef)@residual-score*(j@self.cutoff_coef))/cutoff,
                             response=float(v@self.response_coef), response_grad=j@self.response_coef)
            return cache
        def score(x):
            e=evaluate(x); return e['ratio'], e['ratio_grad']
        def objective(x, sign):
            e=evaluate(x); return sign*e['response'], sign*e['response_grad']
        constraint=dict(type='ineq', fun=lambda x:1-evaluate(x)['ratio'], jac=lambda x:-evaluate(x)['ratio_grad'])
        lo=float('inf'); hi=-float('inf'); xlo=xhi=None; best=float('inf'); bestx=None; results=[]
        counts=dict(score_minimizations=0, endpoint_optimizations=0, optimizer_failures=0, feasible_endpoint_candidates=0)
        for i, start in enumerate(starts):
            res=minimize(score, start, jac=True, bounds=bounds, method='L-BFGS-B',
                         options=dict(maxiter=240, ftol=1e-11, gtol=1e-7))
            counts['score_minimizations']+=1; counts['optimizer_failures']+=int(not res.success)
            e=evaluate(res.x)
            if e['ratio'] < best:
                best=e['ratio']; bestx=res.x.copy()
            if e['ratio'] <= 1+1e-8:
                for sign in (1, -1):
                    opt=minimize(lambda x:objective(x, sign), res.x, jac=True, bounds=bounds,
                                 constraints=constraint, method='SLSQP', options=dict(maxiter=240, ftol=1e-9))
                    counts['endpoint_optimizations']+=1; counts['optimizer_failures']+=int(not opt.success)
                    for xx in (res.x, opt.x):
                        e=evaluate(xx)
                        if e['ratio'] <= 1+1e-7:
                            counts['feasible_endpoint_candidates']+=1
                            if e['response'] < lo:
                                lo=e['response']; xlo=xx.copy()
                            if e['response'] > hi:
                                hi=e['response']; xhi=xx.copy()
            if i+1 in budgets:
                empty=xlo is None
                results.append(dict(starts=i+1, empty=empty, lower=None if empty else lo-self.error,
                                    upper=None if empty else hi+self.error, width=None if empty else hi-lo+2*self.error,
                                    response_error_pp=self.error, best_ratio=best, best_x=bestx.tolist(),
                                    lower_x=None if empty else xlo.tolist(), upper_x=None if empty else xhi.tolist(),
                                    counts=counts.copy()))
        return results

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); subs=parser.add_subparsers(dest='command', required=True)
    train=subs.add_parser('train'); train.add_argument('--root', required=True)
    train.add_argument('--previous', default='results/expanded_pilot'); train.add_argument('--bank', type=int)
    fit=subs.add_parser('fit'); fit.add_argument('--model', required=True); fit.add_argument('--release', required=True)
    fit.add_argument('--output', required=True); fit.add_argument('--starts', type=int, default=32)
    a=parser.parse_args()
    if a.command == 'train':
        for bank in (0, 1) if a.bank is None else (a.bank,):
            print(json.dumps(train_bank(a.root, a.previous, bank)), flush=True)
    else:
        model=RepairedFitter(a.model)
        with np.load(a.release, allow_pickle=False) as z:
            release={k:z[k] for k in z.files}
        result=model.fit(vector(release, model.meta['panel']), budgets=(a.starts,))
        Path(a.output).write_text(json.dumps(dict(model=model.meta, intervals=result), indent=2)+'\n')
