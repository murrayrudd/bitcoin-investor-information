"""Evaluator-only diagnostics. Does not refit, recalibrate, or change intervals."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta,binomtest,t
from scipy.optimize import brentq
from scipy.special import xlogy
from pilot.continuation_design import ROOT,PREVIOUS,PANELS,MAIN_BUDGETS,LARGE_BUDGETS,COUNTS
from pilot.continuation_evaluate import verify
from pilot.repair_fit import RepairedFitter,basis
from pilot.expanded_design import encode
from pilot.expanded_evaluate import wilson

def lower_exact(k,n,alpha=.05):return 0. if k==0 else float(beta.ppf(alpha,k,n-k+1))
def heterogeneous_lower(k,n,alpha=.05):
    """Chernoff lower bound for the mean of independent, unequal Bernoulli rates.

    Concavity gives product_i(1-p_i+p_i*exp(t)) <=
    (1-mean(p)+mean(p)*exp(t))**n. Markov optimization gives exp(-n*KL).
    This statement concerns coverage of the fixed independent reference values.
    """
    if k==0:return 0.
    if k==n:return float(alpha**(1/n))
    phat=k/n
    def f(q):return n*(xlogy(phat,phat/q)+xlogy(1-phat,(1-phat)/(1-q)))-np.log(1/alpha)
    return float(brentq(f,1e-14,phat-1e-14))
def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def run(root=ROOT):
    root=Path(root);verify(root);spec=json.loads((root/'specification.json').read_text());dest=root/'summary';dest.mkdir(exist_ok=True)
    models={(b,p):RepairedFitter(root/'models'/f'bank_{b}_{p}.npz') for b in (0,1) for p in PANELS}
    retention=[];oracles={};oracle_rows=[]
    for c in spec['cases']:
        tid=c['truth_id'];x=encode(c['parameters']);v=basis(x)
        with np.load(root/'oracle'/f'case_{tid:02d}.npz',allow_pickle=False) as z:
            rr=z['response'];mean=float(rr.mean());se=float(rr.std(ddof=1)/np.sqrt(len(rr)));half=float(t.ppf(.995,len(rr)-1)*se)
        oracles[tid]=(mean,half);oracle_rows.append(dict(truth_id=tid,name=c['name'],n=len(rr),mean=mean,mcse=se,half_99=half))
        with np.load(root/'validation'/f'case_{tid:02d}.npz',allow_pickle=False) as z:
            for b in (0,1):
                for p in PANELS:
                    m=models[b,p];y=z[p];scores=np.sum(((y-v@m.coef)@m.white)**2,axis=1)
                    cutoff=float(np.exp(v@m.cutoff_coef));n=len(y);k=int(np.sum(scores<=cutoff));old=int(np.sum(scores<=cutoff/m.meta['cutoff_multiplier']))
                    lo,hi=wilson(k,n)
                    retention.append(dict(bank=b,panel=p,truth_id=tid,name=c['name'],status=c['status'],kind=c['kind'],n=n,accepted=k,
                        retention_pct=100*k/n,old_retention_pct=100*old/n,wilson_low_pct=100*lo,wilson_high_pct=100*hi,
                        exact_one_sided_lower_pct=100*lower_exact(k,n),p_below_target=float(binomtest(k,n,.965,alternative='less').pvalue),
                        point_gate_pass=bool(k/n>=.965),focal_bound_gate_pass=bool(lower_exact(k,n)>=.965)))
    rows=[]
    for path in sorted((root/'fits').glob('*.json')):rows+=json.loads(path.read_text())
    main=[r for r in rows if r['method']=='v0.5' and r['starts']==MAIN_BUDGETS[r['panel']]]
    if len(main)!=len(spec['cases'])*16*8:raise RuntimeError('Incomplete fits')
    search=[];feasibility_failures=[];cell=[];aggregate=[];paired=[]
    lookup={(r['bank'],r['panel'],r['truth_id'],r['replicate'],r['method'],r['starts']):r for r in rows}
    for r in main:
        b,p,tid,rep=r['bank'],r['panel'],r['truth_id'],r['replicate']
        if not r['empty']:
            s=r['seed'];token=hashlib.sha256(f'release-v03-{s}'.encode()).hexdigest()[:16]
            from pilot.inference_features import vector
            with np.load(root/'released/analyst'/f'{token}_{p}.npz',allow_pickle=False) as raw:y=vector({k:raw[k] for k in raw.files},p)
            for side in ('lower','upper'):
                x=np.asarray(r[side+'_x']);q=models[b,p].quantities(x,y)
                if q['ratio']>1+1.0001e-7 or np.any(np.abs(x)>1+1e-8):feasibility_failures.append(dict(bank=b,panel=p,case=tid,replicate=rep,side=side,ratio=q['ratio']))
        checks=[]
        if rep<2:checks.append(('larger_budget',lookup[b,p,tid,rep,'v0.5',LARGE_BUDGETS[p]]))
        if p=='O1' and rep==0:checks.append(('differential_evolution',lookup[b,p,tid,rep,'v0.5_de_check',MAIN_BUDGETS[p]]))
        for method,big in checks:
            reversal=bool(r['empty']!=big['empty']);movement=None
            if not r['empty'] and not big['empty']:
                if big['lower']>r['lower']+1e-10 or big['upper']<r['upper']-1e-10:raise AssertionError('Nonmonotone search')
                movement=max(0.,r['lower']-big['lower'],big['upper']-r['upper'])
            search.append(dict(bank=b,panel=p,truth_id=tid,replicate=rep,method=method,empty_set_reversal=reversal,
                  endpoint_movement_pp=movement,passes=bool(not reversal and (movement is None or movement<=.005))))

    def metrics(rr):
        cover=definite=possible=empty=0;widths=[]
        for r in rr:
            mean,half=oracles[r['truth_id']]
            if r['empty']:empty+=1;continue
            cover+=int(r['lower']<=mean<=r['upper']);definite+=int(r['lower']<=mean-half and mean+half<=r['upper'])
            possible+=int(r['lower']<=mean+half and mean-half<=r['upper']);widths.append(r['width'])
        n=len(rr);lo,hi=wilson(cover,n)
        return dict(n=n,covered=cover,coverage_pct=100*cover/n,coverage_binomial_reference_lower_pct=100*lower_exact(cover,n),
          coverage_heterogeneous_lower_pct=100*heterogeneous_lower(cover,n),
          wilson_low_pct=100*lo,wilson_high_pct=100*hi,definite_coverage_pct=100*definite/n,possible_coverage_pct=100*possible/n,
          empty=empty,empty_pct=100*empty/n,mean_width_pp=float(np.mean(widths)) if widths else None,
          median_width_pp=float(np.median(widths)) if widths else None)

    for b in (0,1):
        for p in PANELS:
            rr=[r for r in main if r['bank']==b and r['panel']==p]
            for scope in ('all','new_location','historical_location_fresh_markets'):
                ss=rr if scope=='all' else [r for r in rr if r['status']==scope]
                aggregate.append(dict(bank=b,panel=p,scope=scope,**metrics(ss)))
            for c in spec['cases']:
                cc=[r for r in rr if r['truth_id']==c['truth_id']]
                cell.append(dict(bank=b,panel=p,truth_id=c['truth_id'],name=c['name'],**metrics(cc)))
        linked={(r['truth_id'],r['replicate']):r for r in main if r['bank']==b and r['panel']=='O6'}
        unlinked={(r['truth_id'],r['replicate']):r for r in main if r['bank']==b and r['panel']=='O7'}
        for scope in ('all','new_location'):
            d=[];n=0
            for key,a in linked.items():
                if scope!='all' and a['status']!=scope:continue
                n+=1;c=unlinked[key]
                if not a['empty'] and not c['empty']:d.append(c['width']-a['width'])
            mean=float(np.mean(d));pooled_se=float(np.std(d,ddof=1)/np.sqrt(len(d)))
            # Delta-method variance of the nonempty-pair ratio, conditional on
            # the fixed locations and fitted models. Empty pairs contribute 0.
            strata={}
            for key,a in linked.items():
                if scope!='all' and a['status']!=scope:continue
                c=unlinked[key];valid=not a['empty'] and not c['empty']
                u=(c['width']-a['width']-mean) if valid else 0.
                strata.setdefault(a['truth_id'],[]).append(u)
            se=float(np.sqrt(sum(len(u)*np.var(u,ddof=1) for u in strata.values()))/len(d))
            h=float(t.ppf(.975,len(d)-1)*se)
            paired.append(dict(bank=b,scope=scope,total_pairs=n,joint_nonempty=len(d),O7_minus_O6_mean_width_pp=mean,paired_mcse_pp=se,
                              pooled_paired_se_pp=pooled_se,ci95_low_pp=mean-h,ci95_high_pp=mean+h,
                              uncertainty_scope='Approximate stratified delta-method Monte Carlo uncertainty conditional on fixed locations and fitted models.'))

    endpoints=json.loads((root/'endpoint_audit/selected_endpoints.json').read_text());audits=[]
    for a in endpoints:
        with np.load(root/'endpoint_audit'/f'endpoint_{a["index"]:03d}.npz',allow_pickle=False) as z:
            response=z['response'];ys=z[a['panel']]
        n=len(response);mean=float(response.mean());se=float(response.std(ddof=1)/np.sqrt(n));half=float(t.ppf(.995,n-1)*se)
        error=abs(mean-a['predicted_response']);m=models[a['bank'],a['panel']];v=basis(a['x'])
        scores=np.sum(((ys-v@m.coef)@m.white)**2,axis=1);cutoff=np.exp(v@m.cutoff_coef)
        audits.append(dict(**a,n=n,response_mean=mean,response_mcse=se,response_half_99=half,absolute_error_pp=error,
             allowance_contains_mean=bool(error<=a['allowance']),allowance_contains_99ci=bool(error+half<=a['allowance']),
             parameter_retention_pct=100*float(np.mean(scores<=cutoff))))

    inventory=[];seen=set();failures=0;maxcash=maxclear=0
    for folder in ('calibration','validation','oracle','endpoint_audit'):
        files=sorted((root/folder).glob('*.npz'));count=0
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                seeds=set(int(s) for s in z['seeds']);meta=json.loads(str(z['metadata']))
                if seen&seeds:raise AssertionError('Duplicate fresh market seeds')
                seen.update(seeds);count+=len(seeds);failures+=len(meta['failures'])
                maxcash=max(maxcash,meta['max_cash']);maxclear=max(maxclear,meta['max_clearing'])
        inventory.append(dict(stage=folder,files=len(files),markets=count))
    release=json.loads((root/'release_reproduction.json').read_text())
    focal=[r for r in retention if r['kind']=='focal_endpoint' and r['panel']=='O6']
    allagg=[r for r in aggregate if r['scope']=='all']
    gates=dict(simulation=bool(failures==0 and max(maxcash,maxclear)<1e-8 and release['all_passed']),
       calibration=bool(all(r['point_gate_pass'] for r in retention) and all(r['focal_bound_gate_pass'] for r in focal)),
       response=bool(all(r['coverage_binomial_reference_lower_pct']>=95 for r in allagg) and all(r['covered']>=15 for r in cell)),
       search=bool(all(r['passes'] for r in search) and not feasibility_failures),
       endpoint_response=bool(all(r['allowance_contains_99ci'] for r in audits)))
    result=dict(version='0.5.0',gates=gates,all_operational_gates_pass=all(gates.values()),
       heterogeneous_response_bound_pass=bool(all(r['coverage_heterogeneous_lower_pct']>=95 for r in allagg)),
       reporting_clarification='The frozen pooled binomial gate is retained as a reference calculation. Because fixed settings can have unequal coverage probabilities, the report additionally uses a conservative independent heterogeneous-Bernoulli Chernoff bound. No inference model, cutoff, interval, simulation allocation, or gate threshold was changed.',
       full_confirmation_authorized=False,decision='Finite-case repair passes; design a separate confirmation protocol.' if all(gates.values()) else 'Keep the large confirmation run on hold; do not retune this validation.',
       inventory=inventory,total_fresh_simulations=len(seen),deterministic_replayed_markets=36*16,failed_simulations=failures,
       release_reproductions=release['exact_reproductions'],max_cash_residual=maxcash,max_clearing_residual=maxclear,
       fitted_market_realizations=36*16,primary_intervals=len(main),calibration_cells=len(retention),search_checks=len(search),
       search_failures=[r for r in search if not r['passes']],max_search_movement_pp=max((r['endpoint_movement_pp'] or 0) for r in search),
       feasibility_failures=feasibility_failures,retention_failures=[r for r in retention if not r['point_gate_pass']],
       focal_bound_failures=[r for r in focal if not r['focal_bound_gate_pass']],endpoint_failures=[r for r in audits if not r['allowance_contains_99ci']],
       min_fixed_case_retention_pct=min(r['retention_pct'] for r in retention),max_endpoint_response_error_pp=max(r['absolute_error_pp'] for r in audits),
       aggregate=allagg,paired=paired,focal=focal,original_confirmation_seeds_used=False)
    for name,rr in [('retention_by_case',retention),('oracle',oracle_rows),('intervals_by_case',cell),('intervals_by_scope',aggregate),('paired_linkage',paired),('search_checks',search),('endpoint_audit',audits)]:write_csv(dest/(name+'.csv'),rr)
    (root/'release_status.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));a=p.parse_args();run(a.root)
