"""Independent verification and numerical stress records for v0.1."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import time
import unittest
import io
import numpy as np
from .config import Config, ENVIRONMENTS
from .economy import simulate


def audit(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parents[1]/"tests"))
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (out/"unit_verification.txt").write_text(log.getvalue())
    record=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    refinement=[]
    for i,(env,change) in enumerate(ENVIRONMENTS.items()):
        cfg=replace(Config(),**change)
        a=simulate(cfg,440000+i)
        b=simulate(replace(cfg,root_grid=257),440000+i)
        refinement.append(dict(environment=env,seed=440000+i,
            max_price_difference=float(np.max(abs(a.prices-b.prices))),
            response_difference=abs(a.response-b.response),
            max_cash18_difference=float(np.max(abs(a.snapshots[18].cash-b.snapshots[18].cash))),
            root_counts_65=[len(d["roots"]) for d in a.diagnostics],
            root_counts_257=[len(d["roots"]) for d in b.diagnostics]))
    stress=[]
    cases={"negative_beliefs":dict(belief_drift=-.03),
           "shrinking_allowances":dict(allowance_growth=-.02),
           "negative_dependence":dict(belief_cash_correlation=-.6),
           "high_commitments":dict(commitment_median=.85),
           "low_risk_aversion":dict(risk_median=.6),
           "high_risk_aversion":dict(risk_median=5.9),
           "mixed_changes":dict(belief_drift=-.03,funding_rate=.01,allowance_growth=-.02),
           "ten_percent_transfer":dict(transfer_share=.10)}
    for i,(name,change) in enumerate(cases.items()):
        try:
            m=simulate(replace(Config(),**change),450000+i)
            stress.append(dict(case=name,seed=450000+i,status="ok",response_percent=m.response,
                max_cash_residual=max(abs(r["cash_ledger_residual"]) for r in m.ledger),
                max_clearing_residual=max(abs(d["clearing_residual"]) for d in m.diagnostics),
                multiple_root_periods=sum(len(d["roots"])>1 or bool(d["zero_spans"]) for d in m.diagnostics)))
        except Exception as exc:
            stress.append(dict(case=name,seed=450000+i,status=type(exc).__name__,error=str(exc)))
    record.update(root_refinement=refinement,stress=stress)
    (out/"verification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(log.getvalue())
    print(json.dumps({"root_refinement":refinement,"stress":stress},indent=2))
    if not result.wasSuccessful():raise RuntimeError("Unit verification failed")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="results/verification")
    audit(p.parse_args().output)
