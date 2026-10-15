"""Fixed summaries computed solely from a released observation panel.

O6 and O7 share exactly the same marginal summaries. O6 alone adds
cross-module links. Longitudinal and within-module links remain in O7.
"""
import numpy as np


def _corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.std()<1e-12 or b.std()<1e-12:return 0.0
    return float(np.corrcoef(a,b)[0,1])


def summaries(data,panel):
    if str(data['panel'])!=panel:raise ValueError('Panel label mismatch')
    if panel not in ('O1','O2','O6','O7'):raise ValueError(panel)
    out={}
    for i in (0,5,11,17):
        out[f'log_price_{i+1}']=float(np.log(data['prices'][i]))
        out[f'signal_{i+1}']=float(data['signal'][i])
    # 0.05 is the known initial per-person stock/cash in the frozen N=2000 design.
    scale=.05
    h=np.asarray(data['holdings'])
    target=data['target_holdings'] if panel=='O7' else data['target']
    for wave,month in ((0,6),(2,18)):
        out[f'hold_median_{month}']=float(np.median(np.log1p(h[wave]/scale)))
        out[f'hold_q90_{month}']=float(np.quantile(np.log1p(h[wave]/scale),.9))
        out[f'hold_zero_{month}']=float(np.mean(h[wave]<1e-10))
    delta=(h[2]-h[0])/scale
    out['hold_change_q10']=float(np.quantile(delta,.1))
    out['hold_change_q90']=float(np.quantile(delta,.9))
    out['target_hold_mean_18']=float(np.mean(h[2][target[2]])/scale)
    if panel in ('O2','O6','O7'):
        q=np.asarray(data['belief'])
        for wave,month in ((0,6),(2,18)):
            out[f'belief_mean_{month}']=float(q[wave].mean())
            out[f'belief_sd_{month}']=float(q[wave].std())
    if panel in ('O6','O7'):
        cash=np.asarray(data['cash']);floor=np.asarray(data['commitment'])
        usable=cash-floor
        cap=np.asarray(data['cap']);finite=np.isfinite(cap[0])
        for wave,month in ((0,6),(2,18)):
            out[f'cash_logmean_{month}']=float(np.log1p(cash[wave]/scale).mean())
            out[f'usable_mean_{month}']=float(usable[wave].mean()/scale)
            out[f'cap_logmedian_{month}']=float(np.median(np.log1p(cap[wave,finite]/scale)))
            out[f'cap_logq90_{month}']=float(np.quantile(np.log1p(cap[wave,finite]/scale),.9))
        out['floor_logmean']=float(np.log1p(floor[0]/scale).mean())
        out['unrestricted_share']=float(np.asarray(data['unrestricted'])[0].mean())
        out['risk_ce_mean']=float(np.asarray(data['risk_ce'])[0].mean())
        out['risk_ce_sd']=float(np.asarray(data['risk_ce'])[0].std())
    if panel in ('O2','O6'):
        out['linked_belief_holdings']=_corr(q[2],np.log1p(h[2]/scale))
    if panel=='O6':
        out['linked_belief_cash']=_corr(q[2],np.log1p(cash[2]/scale))
        out['linked_belief_usable']=_corr(q[2],np.log1p(usable[2]/scale))
        out['linked_cash_holdings']=_corr(np.log1p(cash[2]/scale),np.log1p(h[2]/scale))
        out['linked_risk_holdings']=_corr(data['risk_ce'][0],np.log1p(h[2]/scale))
        out['linked_cap_active']=float(np.mean(np.isfinite(cap[2]) & (abs(h[2]-cap[2])<1e-8)))
    values=np.asarray(list(out.values()),float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite released summary')
    return out


def vector(data,panel):
    return np.asarray(list(summaries(data,panel).values()),float)
