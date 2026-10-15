"""Generate data and feasibility diagnostics, not confidence intervals.

Usage: python -m pilot.run --replications 50 --output results/full
"""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import scipy
from .config import Config, ENVIRONMENTS
from .economy import simulate, EconomicInfeasibility, NumericalFailure
from .observations import release_panels, SamplingFailure, PANELS, REGIMES
from .features import summaries


def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def save_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(output,replications=50,n=2000,sample_size=250,root_grid=65,stage="pilot",store_data=True):
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory; preserve previous runs")
    out.mkdir(parents=True,exist_ok=True)
    for name in ["analyst","evaluation","benchmarks"]:
        (out/name).mkdir()
    config=Config(n=n,sample_size=sample_size,root_grid=root_grid).validate()
    bases={"development":110000,"pilot":220000,"confirmation":330000}
    if stage=="confirmation":
        raise ValueError("Confirmation inference is not implemented in v0.1")
    base=bases[stage]
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(Path(__file__).parent.glob("*.py"))}
    save_json(out/"specification.json",dict(version="0.1.0",stage=stage,
        config=asdict(config),environments=ENVIRONMENTS,replications=replications,
        seed_rule="base + 10000*environment_index + replication; observation base + 1000000",
        seed_base=base,source_sha256=source_hashes,
        estimator_status="not implemented; generation and observation feasibility only"))
    rows=[];datasets=[];features=[];ledgers=[];diagnostics=[];failures=[]
    start=time.perf_counter()
    for ei,(environment,change) in enumerate(ENVIRONMENTS.items()):
        cfg=replace(config,**change)
        for rep in range(replications):
            seed=base+10000*ei+rep
            token=hashlib.sha256(f"market-v01-{seed}".encode()).hexdigest()[:16]
            tic=time.perf_counter()
            try:
                market=simulate(cfg,seed)
                elapsed=time.perf_counter()-tic
                final=[d for d in market.diagnostics if d["branch"]=="control" and d["month"]==18][0]
                p=market.population
                row=dict(market_id=token,environment=environment,replication=rep,seed=seed,
                         status="ok",runtime_seconds=elapsed,price_1=float(market.prices[0]),
                         price_18=float(market.prices[17]),response_percent=market.response,
                         response_six_percent=market.response_six,transfer=market.transfer,
                         cash_binding_18=final["cash_binding"],cap_binding_18=final["cap_binding"],
                         joint_binding_18=final["joint_binding"],zero_position_18=final["zero_position"],
                         max_cash_residual=max(abs(d["cash_ledger_residual"]) for d in market.ledger),
                         max_clearing_residual=max(abs(d["clearing_residual"]) for d in market.diagnostics),
                         max_kkt=max(d["kkt_violation"] for d in market.diagnostics),
                         multiple_root_periods=sum(len(d["roots"])>1 or bool(d["zero_spans"]) for d in market.diagnostics),
                         realized_belief_cash_correlation=float(np.corrcoef(p.belief_intercept,p.cash0)[0,1]),
                         realized_belief_usable_cash_correlation=float(np.corrcoef(p.belief_intercept,p.cash0-p.floor)[0,1]))
                rows.append(row)
                for r in market.ledger:ledgers.append(dict(market_id=token,**r))
                for r in market.diagnostics:diagnostics.append(dict(market_id=token,**r))
                # One complete checkpoint per environment. Remaining truth is
                # reproducible from the frozen config and evaluator-only seed.
                if rep==0:
                    np.savez_compressed(out/"benchmarks"/f"{environment}_checkpoint.npz",
                        cash0=p.cash0,holdings0=p.holdings0,floor=p.floor,gamma=p.gamma,
                        belief_intercept=p.belief_intercept,allowance0=p.allowance0,
                        unrestricted=p.unrestricted,target=p.target,signal=market.signal,
                        prices=market.prices,cash18=market.snapshots[18].cash,
                        holdings18=market.snapshots[18].holdings,
                        response=np.array(market.response))
                for regime in REGIMES:
                    try:
                        panels,selection=release_panels(market,cfg,seed+1000000,regime)
                    except SamplingFailure as exc:
                        for panel in PANELS:
                            failures.append(dict(market_id=token,environment=environment,
                                regime=regime,panel=panel,status="sampling_failure",error=str(exc)))
                        continue
                    for panel,data in panels.items():
                        release_token=hashlib.sha256(f"release-v01-{seed}-{regime}".encode()).hexdigest()[:16]
                        dataset_id=f"{release_token}_{panel}"
                        if store_data:
                            np.savez_compressed(out/"analyst"/f"{dataset_id}.npz",**data)
                        feat=summaries(data,panel)
                        features.append(dict(dataset_id=dataset_id,panel=panel,**feat))
                        datasets.append(dict(dataset_id=dataset_id,market_id=token,environment=environment,
                            panel=panel,regime=regime,status="ok",records=cfg.sample_size*len(cfg.survey_months),
                            **selection))
            except Exception as exc:
                status="economic_infeasibility" if isinstance(exc,EconomicInfeasibility) else "numerical_failure" if isinstance(exc,NumericalFailure) else "unexpected_failure"
                rows.append(dict(market_id=token,environment=environment,replication=rep,seed=seed,status=status))
                failures.append(dict(market_id=token,environment=environment,status=status,error=str(exc),
                                     traceback=traceback.format_exc()))
            if (rep+1)%10==0 or rep==0:
                print(f"{environment}: {rep+1}/{replications} markets; {len(failures)} failure records",flush=True)
    save_csv(out/"evaluation"/"markets.csv",rows)
    save_csv(out/"evaluation"/"datasets.csv",datasets)
    save_csv(out/"evaluation"/"cash_ledger.csv",ledgers)
    save_csv(out/"analyst"/"features.csv",features)
    save_json(out/"evaluation"/"clearing_diagnostics.json",diagnostics)
    save_json(out/"evaluation"/"failures.json",failures)
    save_json(out/"runtime.json",dict(seconds=time.perf_counter()-start,python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        attempted_markets=3*replications,successful_markets=sum(r["status"]=="ok" for r in rows),
        attempted_datasets=3*replications*len(PANELS)*len(REGIMES),successful_datasets=len(datasets),
        inference_evaluations=0))
    print(json.dumps(json.loads((out/"runtime.json").read_text()),indent=2),flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replications",type=int,default=50)
    p.add_argument("--population",type=int,default=2000)
    p.add_argument("--sample-size",type=int,default=250)
    p.add_argument("--root-grid",type=int,default=65)
    p.add_argument("--stage",choices=["development","pilot"],default="pilot")
    p.add_argument("--output",required=True)
    p.add_argument("--no-data",action="store_true")
    args=p.parse_args()
    if args.replications<=0: p.error("replications must be positive")
    run(args.output,args.replications,args.population,args.sample_size,args.root_grid,args.stage,not args.no_data)


if __name__=="__main__":main()
