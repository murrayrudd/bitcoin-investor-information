"""Observation release. Analyst files contain neither truth nor future data."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

PANELS=("O1","O2","O6","O7")
REGIMES=("clean","selective")


class SamplingFailure(RuntimeError):
    pass


def risk_task_ce(gamma):
    """Certainty equivalent for equal chances of 0.5 and 1.5 wealth units."""
    a=1-gamma
    out=np.empty_like(a)
    islog=abs(a)<1e-8
    out[islog]=np.sqrt(.5*1.5)
    out[~islog]=np.exp(logsumexp(np.log([.5,1.5])[:,None]*a[None,~islog],axis=0)
                         /a[~islog]-np.log(2)/a[~islog])
    return out


def select_sample(market,cfg,seed,regime):
    rng=np.random.default_rng(seed)
    q=market.snapshots[cfg.survey_months[0]].belief
    score=(q-np.median(q))/max(float(np.subtract(*np.percentile(q,[75,25]))),1e-12)
    slope=0.0 if regime=="clean" else np.log(cfg.selection_odds_ratio)
    if regime not in REGIMES:
        raise ValueError(regime)
    intercept=brentq(lambda a: expit(a+slope*score).mean()-cfg.participation_mean,-30,30)
    inclusion=expit(intercept+slope*score)
    eligible=np.flatnonzero(rng.uniform(size=cfg.n)<inclusion)
    if len(eligible)<cfg.sample_size:
        raise SamplingFailure("Insufficient respondents; no replacement market is drawn")
    chosen=rng.choice(eligible,cfg.sample_size,replace=False)
    return chosen,dict(eligible=int(len(eligible)),participation_min=float(inclusion.min()),
                       participation_max=float(inclusion.max()),
                       population_belief=float(q.mean()),sample_belief=float(q[chosen].mean()))


def release_panels(market,cfg,seed,regime):
    idx,selection=select_sample(market,cfg,seed,regime)
    common=dict(months=np.asarray(cfg.survey_months),prices=market.prices[:cfg.cutoff].copy(),
                signal=market.signal[:cfg.cutoff].copy(),initial_cash=np.array(cfg.initial_cash),
                bitcoin_stock=np.array(cfg.stock))
    snapshots=[market.snapshots[t] for t in cfg.survey_months]
    fields=dict(holdings=np.stack([s.holdings[idx] for s in snapshots]),
                belief=np.stack([s.belief[idx] for s in snapshots]),
                cash=np.stack([s.cash[idx] for s in snapshots]),
                commitment=np.tile(market.population.floor[idx],(len(snapshots),1)),
                cap=np.stack([s.cap[idx] for s in snapshots]),
                unrestricted=np.tile(market.population.unrestricted[idx],(len(snapshots),1)),
                risk_ce=np.tile(risk_task_ce(market.population.gamma[idx]),(len(snapshots),1)))
    result={}
    target=np.tile(market.population.target[idx],(len(snapshots),1))
    result["O1"]={**common,"holdings":fields["holdings"].copy(),"target":target.copy()}
    result["O2"]={**common,**{k:fields[k].copy() for k in ("holdings","belief")},"target":target.copy()}
    result["O6"]={**common,**{k:v.copy() for k,v in fields.items()},"target":target.copy()}
    # Preserve longitudinal links and within-resource/permission links. Remove
    # every cross-module person link; array column numbers are NOT common IDs.
    rng=np.random.default_rng(np.random.SeedSequence([seed,703]))
    unlinked={}
    for label,group in [("holdings",("holdings",)),("belief",("belief",)),
                        ("resources",("cash","commitment")),("permissions",("cap","unrestricted")),
                        ("risk",("risk_ce",))]:
        order=rng.permutation(cfg.sample_size)
        for k in group:
            unlinked[k]=fields[k][:,order].copy()
        unlinked[f"target_{label}"]=target[:,order].copy()
    result["O7"]={**common,**unlinked}
    for panel,data in result.items():
        data["panel"]=np.array(panel)
    return result,selection
