"""Independent validation; generating parameters are used only for scoring.

The default fit never receives the true parameter. A separately labeled
known-participation diagnostic receives only that one stipulated parameter.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import t
from .expanded_design import truths,encode,COUNTS,SEARCH_TOLERANCE
from .expanded_fit import ContinuousFitter,basis
from .observations import PANELS

MODELS={}
def init(root):
    for b in range(2):
        for p in PANELS:MODELS[b,p]=ContinuousFitter(Path(root)/'models'/f'bank_{b}_{p}.npz')

def evaluate_job(job):
    root=Path(job['root']);b=job['bank'];p=job['panel'];case=job['case'];tid=case['truth_id']
    model=MODELS[b,p];start=time.perf_counter();xtrue=encode(case['parameters']);rows=[]
    with np.load(root/'evaluation'/f'truth_{tid:02d}.npz') as z:features=z[p];seeds=z['seeds']
    for r,y in enumerate(features):
        budgets=(2,8,24) if r<5 else (2,8)
        fits=model.fit(y,budgets=budgets)
        true_score=float(np.sum((basis(xtrue)@model.wcoef-y@model.white)**2))
        for fit in fits:
            rows.append(dict(bank=b,panel=p,truth_id=tid,name=case['name'],kind=case['kind'],replicate=r,seed=int(seeds[r]),
              participation='unknown',true_parameter_accepted=true_score<=model.cutoff,true_score=true_score,**fit))
        if p=='O6' and r<10:
            fit=model.fit(y,budgets=(8,),fixed_selection=float(xtrue[-1]))[0]
            rows.append(dict(bank=b,panel=p,truth_id=tid,name=case['name'],kind=case['kind'],replicate=r,seed=int(seeds[r]),
              participation='known_diagnostic',true_parameter_accepted=true_score<=model.cutoff,true_score=true_score,**fit))
    dest=root/'fits';dest.mkdir(exist_ok=True);path=dest/f'bank_{b}_{p}_truth_{tid:02d}.json'
    path.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    return dict(bank=b,panel=p,truth_id=tid,rows=len(rows),seconds=time.perf_counter()-start)

def run(root,workers=8):
    root=Path(root);(root/'fits').mkdir(exist_ok=True);pending=[]
    for b in range(2):
        for p in PANELS:
            for case in truths():
                if not (root/'fits'/f'bank_{b}_{p}_truth_{case["truth_id"]:02d}.json').exists():
                    pending.append(dict(root=str(root),bank=b,panel=p,case=case))
    start=time.perf_counter();logs=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(str(root),)) as pool:
        fs=[pool.submit(evaluate_job,j) for j in pending]
        for i,f in enumerate(as_completed(fs)):
            result=f.result();logs.append(result)
            progress=dict(done=i+1,total=len(fs),elapsed=time.perf_counter()-start,last=result)
            print(json.dumps(progress),flush=True);(root/'progress_fitting.json').write_text(json.dumps(progress,indent=2)+'\n')
    summarize(root)

def wilson(k,n):
    z=1.959963984540054;p=k/n;center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(rows,oracle):
    cover=[];definite=[];possible=[];width=[];accept=[];empty=[]
    for r in rows:
        o=oracle[r['truth_id']];lo=r['lower'];hi=r['upper'];e=r['empty'];m=o['mean'];h=o['half_99']
        empty.append(e);accept.append(r['true_parameter_accepted'])
        cover.append(not e and lo<=m<=hi)
        definite.append(not e and lo<=m-h and hi>=m+h)
        possible.append(not e and lo<=m+h and hi>=m-h)
        if not e:width.append(hi-lo)
    n=len(rows);wl,wu=wilson(sum(cover),n)
    return dict(n=n,coverage_pct=100*np.mean(cover),wilson_low_pct=100*wl,wilson_high_pct=100*wu,
      oracle_certain_coverage_pct=100*np.mean(definite),oracle_possible_coverage_pct=100*np.mean(possible),
      empty_pct=100*np.mean(empty),true_parameter_acceptance_pct=100*np.mean(accept),
      mean_width_pp=float(np.mean(width)) if width else None,median_width_pp=float(np.median(width)) if width else None)

def summarize(root):
    root=Path(root);dest=root/'summary';dest.mkdir(exist_ok=True)
    rows=[]
    for p in sorted((root/'fits').glob('*.json')):rows.extend(json.loads(p.read_text()))
    oracle={}
    for case in truths():
        i=case['truth_id']
        with np.load(root/'oracle'/f'truth_{i:02d}.npz') as z:v=z['response']
        se=float(v.std(ddof=1)/np.sqrt(len(v)))
        oracle[i]=dict(truth_id=i,name=case['name'],kind=case['kind'],n=len(v),mean=float(v.mean()),mcse=se,half_99=float(t.ppf(.995,len(v)-1)*se))
    write_csv(dest/'oracle.csv',list(oracle.values()))
    main=[r for r in rows if r['participation']=='unknown' and r['starts']==8]
    out=[];cells=[]
    for b in range(2):
        for p in PANELS:
            for kind in ['all','pure','mixed','boundary']:
                selected=[r for r in main if r['bank']==b and r['panel']==p and (kind=='all' or r['kind']==kind)]
                out.append(dict(bank=b,panel=p,kind=kind,**stats(selected,oracle)))
            for case in truths():
                selected=[r for r in main if r['bank']==b and r['panel']==p and r['truth_id']==case['truth_id']]
                cells.append(dict(bank=b,panel=p,truth_id=case['truth_id'],name=case['name'],kind=case['kind'],**stats(selected,oracle)))
    write_csv(dest/'by_scope.csv',out);write_csv(dest/'by_case.csv',cells)
    pair=[];bankdiff=[];search=[];known=[]
    for b in range(2):
        for kind in ['all','pure','mixed','boundary']:
            subset=[r for r in main if r['bank']==b and (kind=='all' or r['kind']==kind)]
            linked={(r['truth_id'],r['replicate']):r for r in subset if r['panel']=='O6'}
            unlinked={(r['truth_id'],r['replicate']):r for r in subset if r['panel']=='O7'}
            dif=[];covdif=[]
            for key,a in linked.items():
                c=unlinked[key];m=oracle[a['truth_id']]['mean']
                ca=not a['empty'] and a['lower']<=m<=a['upper'];cc=not c['empty'] and c['lower']<=m<=c['upper']
                covdif.append(int(ca)-int(cc))
                if not a['empty'] and not c['empty']:dif.append(c['width']-a['width'])
            pair.append(dict(bank=b,kind=kind,n_pairs=len(linked),n_joint_nonempty=len(dif),mean_O7_minus_O6_width_pp=float(np.mean(dif)),
              paired_width_se_pp=float(np.std(dif,ddof=1)/np.sqrt(len(dif))),coverage_O6_minus_O7_pp=float(100*np.mean(covdif)),
              paired_coverage_se_pp=float(100*np.std(covdif,ddof=1)/np.sqrt(len(covdif)))))
        for p in PANELS:
            index={(r['truth_id'],r['replicate'],r['starts']):r for r in rows if r['bank']==b and r['panel']==p and r['participation']=='unknown'}
            for small,large in [(2,8),(8,24)]:
                gains=[];moves=[];resolved=0;n=0;shrunk=0
                for (tid,rep,starts),big in index.items():
                    if starts!=large:continue
                    a=index[tid,rep,small];n+=1
                    if a['empty'] and not big['empty']:resolved+=1
                    if not a['empty'] and not big['empty']:
                        gain=big['width']-a['width'];gains.append(gain)
                        moves.append(max(a['lower']-big['lower'],big['upper']-a['upper']))
                        shrunk+=gain < -1e-9
                search.append(dict(bank=b,panel=p,small_starts=small,large_starts=large,n_pairs=n,new_nonempty=resolved,shrinking_intervals=shrunk,
                  median_width_gain_pp=float(np.median(gains)),p95_width_gain_pp=float(np.quantile(gains,.95)),max_width_gain_pp=float(np.max(gains)),
                  endpoint_change_above_tolerance_pct=float(100*np.mean(np.asarray(moves)>SEARCH_TOLERANCE)),max_endpoint_change_pp=float(np.max(moves))))
        for r in rows:
            if r['bank']==b and r['participation']=='known_diagnostic':
                a=next(x for x in main if x['bank']==b and x['panel']=='O6' and x['truth_id']==r['truth_id'] and x['replicate']==r['replicate'])
                known.append(dict(bank=b,truth_id=r['truth_id'],replicate=r['replicate'],known_empty=r['empty'],unknown_empty=a['empty'],
                  known_width=r['width'],unknown_width=a['width'],width_cost=None if r['empty'] or a['empty'] else a['width']-r['width']))
    for p in PANELS:
        a={(r['truth_id'],r['replicate']):r for r in main if r['bank']==0 and r['panel']==p}
        b={(r['truth_id'],r['replicate']):r for r in main if r['bank']==1 and r['panel']==p}
        d=[]
        for k in a:
            if not a[k]['empty'] and not b[k]['empty']:d.append(max(abs(a[k]['lower']-b[k]['lower']),abs(a[k]['upper']-b[k]['upper'])))
        bankdiff.append(dict(panel=p,n_pairs=len(a),joint_nonempty=len(d),mean_max_endpoint_difference_pp=float(np.mean(d)),
           p95_max_endpoint_difference_pp=float(np.quantile(d,.95)),max_endpoint_difference_pp=float(np.max(d))))
    write_csv(dest/'linked_vs_unlinked.csv',pair);write_csv(dest/'search_stability.csv',search)
    write_csv(dest/'independent_banks.csv',bankdiff);write_csv(dest/'participation_cost.csv',known)
    flat=[]
    for r in rows:
        scalar={k:v for k,v in r.items() if k not in ('best_x','lower_x','upper_x','counts')}
        scalar.update({f'count_{k}':v for k,v in r['counts'].items()});flat.append(scalar)
    write_csv(dest/'fits.csv',flat)
    (dest/'summary_counts.json').write_text(json.dumps(dict(total_rows=len(rows),main_intervals=len(main),validation_markets=12*COUNTS['evaluation_reps'],independent_fitted_banks=2,
      note='Banks share validation markets; do not count their estimates as independent repetitions. Widths exclude empty sets; empties count as noncoverage.'),indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--summary-only',action='store_true');a=p.parse_args()
    summarize(a.root) if a.summary_only else run(a.root,a.workers)
