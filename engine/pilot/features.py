"""Candidate inference summaries, operating ONLY on released observations.

There is no model, truth-record, or intervention import in this module.
These summaries are an interface, not a calibrated confidence procedure.
"""
import numpy as np
from scipy.stats import spearmanr


def summaries(data,panel):
    if str(data["panel"]) != panel:
        raise ValueError("Panel metadata disagree; cross-module links cannot be invented")
    out={}
    for t in [1,6,12,18]:
        if t<=len(data["prices"]):
            out[f"price_{t}"]=float(data["prices"][t-1])
    for field in ["holdings","belief","cash","commitment","cap","risk_ce"]:
        if field not in data:
            continue
        for j,month in enumerate(data["months"]):
            x=data[field][j]
            finite=x[np.isfinite(x)]
            for pct in [10,50,90]:
                out[f"{field}_{month}_q{pct}"]=float(np.percentile(finite,pct)) if len(finite) else None
        if field=="cap":
            out["unrestricted_fraction"]=float(data["unrestricted"][-1].mean())
    if panel in ("O2","O6"):
        out["belief_holdings_rank_correlation"]=float(spearmanr(data["belief"][-1],data["holdings"][-1]).statistic)
    if panel=="O6":
        available=data["cash"][-1]-data["commitment"][-1]
        out["belief_available_cash_rank_correlation"]=float(spearmanr(data["belief"][-1],available).statistic)
        out["belief_cash_rank_correlation"]=float(spearmanr(data["belief"][-1],data["cash"][-1]).statistic)
    # The intervention group is observable. Binary group labels are retained
    # within O7 modules and do not restore person-level cross-module joins.
    for k in ["holdings","belief","cash","risk_ce"]:
        if k not in data: continue
        label={"cash":"resources","risk_ce":"risk"}.get(k,k)
        target=data[f"target_{label}"][-1] if panel=="O7" else data["target"][-1]
        out[f"target_mean_{k}"]=float(data[k][-1,target].mean()) if target.any() else None
    if panel=="O7" and any("correlation" in k for k in out):
        raise AssertionError("O7 cross-module linkage leaked")
    return out
