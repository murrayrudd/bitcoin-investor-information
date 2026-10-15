"""Join final numerical benchmarks and verify the added follow-up simulations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from pilot.repair_evaluate import verify_freeze
from pilot.expanded_evaluate import write_csv

def read(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def run(root):
    root=Path(root);verify_freeze(root);s=root/'summary'
    rows=[r for r in read(s/'by_scope.csv') if r['panel']!='O1']
    rows+=read(s/'holdings_search_by_scope.csv')
    rows.sort(key=lambda r:(int(r['bank']),r['scope'],r['panel']))
    write_csv(s/'final_by_scope.csv',rows)
    extension=json.loads((root/'holdings_search_freeze.json').read_text())
    source=Path(__file__).resolve().parents[1]/extension['source']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=extension['source_sha256']:
        raise AssertionError('Expanded search source changed')
    folders=[root/f'bank_{b}'/role for b in (0,1) for role in ('train','check','quantile','calibration','response')]
    folders += [root/'evaluation',root/'oracle',root/'endpoint_audit',root/'endpoint_followup']
    seen=set();count=followup=0;clear=cash=kkt=0.
    for folder in folders:
        for path in folder.glob('*.npz'):
            with np.load(path,allow_pickle=False) as z:
                if 'metadata' not in z:continue
                meta=json.loads(str(z['metadata']));raw=list(map(int,z['seeds']));seeds=set(raw)
                if len(seeds)!=len(raw) or seen&seeds or meta['failures'] or meta['successful']!=meta['attempted']:
                    raise AssertionError(f'Invalid or overlapping simulations: {path}')
                if min(seeds)<600_000_000 or max(seeds)>=800_000_000:
                    raise AssertionError('Unexpected seed namespace')
                seen.update(seeds);count+=meta['successful']
                if folder.name=='endpoint_followup':followup+=meta['successful']
                clear=max(clear,meta['max_clearing']);cash=max(cash,meta['max_cash']);kkt=max(kkt,meta['max_kkt'])
    if seen&{v+2_000_000_000 for v in seen}:raise AssertionError('Observation seed collision')
    principal=json.loads((s/'run_audit.json').read_text())
    if count!=principal['fresh_markets']+1536 or followup!=1536:
        raise AssertionError('Unexpected total market count')
    checks=read(s/'holdings_search_checks.csv')
    follow=json.loads((root/'endpoint_followup/results.json').read_text())
    result=dict(version='0.4.0',statistical_source_and_models_unchanged=True,
        final_O1_starts=96,other_panels_starts=12,
        final_summary='summary/final_by_scope.csv',
        total_fresh_markets_including_followup=count,unique_market_seeds=len(seen),
        simulation_failures=0,max_clearing=clear,max_cash=cash,max_kkt=kkt,
        original_confirmation_seeds_used=False,ready_for_parameter_conditional_confirmation=False,
        reason='Two independently repeated selected locations retain parameters below the 96.5% target; seven of 288 O1 checks still move materially from 96 to 192 starts. Marginal calibration does not establish conditional coverage.',
        interpretation='Finite-case validation and approximate component-marginal inference; no uniform parameter-conditional guarantee.',
        O1_96_to_192_checks=[r for r in checks if int(r['small'])==96],
        followup=follow)
    (s/'release_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('followup','O1_96_to_192_checks')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/repaired_pilot');a=p.parse_args();run(a.root)
