"""Descriptive summaries of generator behavior, separate from inference tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .run import save_json, save_csv


def mean_se(x):
    a=np.asarray(x,float)
    return dict(n=len(a),mean=float(a.mean()),mcse=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,
                minimum=float(a.min()),maximum=float(a.max()))


def summarize(root):
    root=Path(root);out=root/"summary";out.mkdir(exist_ok=True)
    markets=list(csv.DictReader((root/"evaluation/markets.csv").open()))
    data=list(csv.DictReader((root/"evaluation/datasets.csv").open()))
    records=[];table=[]
    for env in ["belief","funding","mandate"]:
        rows=[r for r in markets if r["environment"]==env and r["status"]=="ok"]
        fields={k:mean_se([r[k] for r in rows]) for k in ["price_1","price_18","response_percent",
            "response_six_percent","cash_binding_18","cap_binding_18","joint_binding_18","zero_position_18",
            "realized_belief_usable_cash_correlation"]}
        selection={regime:mean_se([float(d["sample_belief"])-float(d["population_belief"])
            for d in data if d["environment"]==env and d["regime"]==regime and d["panel"]=="O6"])
            for regime in ["clean","selective"]}
        records.append(dict(environment=env,fields=fields,selection_bias=selection))
        table.append(dict(environment=env,markets=len(rows),
            mean_response_percent=fields["response_percent"]["mean"],
            mcse_response_percentage_points=fields["response_percent"]["mcse"],
            mean_cash_binding_percent=100*fields["cash_binding_18"]["mean"],
            mean_cap_binding_percent=100*fields["cap_binding_18"]["mean"],
            clean_belief_sample_bias_pp=100*selection["clean"]["mean"],
            selective_belief_sample_bias_pp=100*selection["selective"]["mean"]))
    ok=[r for r in markets if r["status"]=="ok"]
    summary=dict(environments=records,successful_markets=len(ok),successful_datasets=len(data),
        max_cash_residual=max(float(r["max_cash_residual"]) for r in ok),
        max_clearing_residual=max(float(r["max_clearing_residual"]) for r in ok),
        max_kkt=max(float(r["max_kkt"]) for r in ok),
        multiple_root_periods=sum(int(r["multiple_root_periods"]) for r in ok),
        status="Generator feasibility; no estimator coverage or measurement advantage has been evaluated")
    save_json(out/"feasibility.json",summary)
    save_csv(out/"environment_summary.csv",table)
    # Explicitly audit identical marginal and within-module group information
    # in O6/O7 for every released pair; no cross-module join is attempted.
    pairs=0;max_group_error=0.0
    for p in sorted((root/"analyst").glob("*_O6.npz")):
        q=p.with_name(p.name.replace("_O6.npz","_O7.npz"))
        with np.load(p) as a,np.load(q) as b:
            for k in ["holdings","belief","cash","commitment","cap","risk_ce"]:
                if not np.array_equal(np.sort(a[k],axis=1),np.sort(b[k],axis=1)):
                    raise AssertionError(f"Marginals differ: {p.name} {k}")
            for field,module in [("holdings","holdings"),("belief","belief"),("cash","resources"),
                                 ("commitment","resources"),("cap","permissions"),("risk_ce","risk")]:
                for j in range(3):
                    av=np.sort(a[field][j,a["target"][j]])
                    bv=np.sort(b[field][j,b[f"target_{module}"][j]])
                    if not np.array_equal(av,bv):raise AssertionError("Group marginals differ")
        pairs+=1
    save_json(out/"observation_audit.json",dict(pairs_checked=pairs,all_marginals_identical=True,
        target_group_marginals_identical=True,linked_cross_module_moments_available_only_for_O6=True))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.linewidth":.8,"savefig.dpi":220})
    fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout="constrained")
    x=np.arange(3);labels=["Belief change","Additional funding","Mandate relaxation"]
    y=[r["mean_response_percent"] for r in table]
    err=[r["mcse_response_percentage_points"] for r in table]
    axs[0].errorbar(x,y,yerr=err,fmt="o",color="black",capsize=4,lw=1.2,markersize=5)
    axs[0].set_xticks(x,labels,rotation=15,ha="right")
    axs[0].set_ylabel("Price response to cash transfer (%)")
    axs[0].set_title("A  Mean response and Monte Carlo SE",loc="left",fontweight="bold",fontsize=10)
    axs[0].set_ylim(.85,1.35)
    axs[0].grid(axis="y",color=".85",linewidth=.5)
    axs[1].bar(x-.18,[r["mean_cash_binding_percent"] for r in table],.34,label="Cash commitment",color=".35",edgecolor="black",lw=.6)
    axs[1].bar(x+.18,[r["mean_cap_binding_percent"] for r in table],.34,label="Position ceiling",color="white",edgecolor="black",hatch="///",lw=.6)
    axs[1].set_xticks(x,labels,rotation=15,ha="right")
    axs[1].set_ylabel("Investors with binding restriction (%)")
    axs[1].set_ylim(0,36)
    axs[1].set_title("B  Constraints at month 18",loc="left",fontweight="bold",fontsize=10)
    axs[1].legend(frameon=False,fontsize=9)
    for ext in ["png","svg"]:fig.savefig(out/f"generator_diagnostics.{ext}")
    plt.close(fig)
    print(json.dumps({"summary":table,"audit":{k:v for k,v in summary.items() if k!="environments"}},indent=2))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory")
    summarize(p.parse_args().directory)
