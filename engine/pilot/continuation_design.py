"""Prespecified single-pass continuation from the unchanged version 0.4 models."""
import json
from pathlib import Path
import numpy as np
from .expanded_design import encode, decode
from .repair_design import cases as old_cases, draw

VERSION = '0.5.0'
PANELS = ('O1', 'O2', 'O6', 'O7')
COUNTS = dict(anchors_per_bank=12, calibration_reps=512, validation_reps=512,
              fitted_reps=16, oracle_reps=512, audit_reps=128)
MAIN_BUDGETS = dict(O1=64, O2=24, O6=24, O7=24)
LARGE_BUDGETS = dict(O1=128, O2=48, O6=48, O7=48)
PREVIOUS = Path('results/repaired_pilot')
ROOT = Path('results/continuation_v05')

def seed(bank, role, index, rep):
    roles = dict(calibration=0, validation=1, oracle=2, audit=3)
    if bank not in (0, 1) or role not in roles or not 0 <= index < 1000 or not 0 <= rep < 10000:
        raise ValueError('Seed coordinate outside allocation')
    return 10_000_000_000 + bank*100_000_000 + roles[role]*10_000_000 + index*10000 + rep

def anchors(previous=PREVIOUS):
    audit=json.loads((Path(previous)/'endpoint_audit/audit_results.json').read_text())
    out=[]
    for bank in (0, 1):
        candidates=[r for r in audit if r['bank']==bank and not r['empty']]
        ranked=sorted(candidates, key=lambda r:(r['parameter_acceptance_pct'], r['index']))
        chosen=ranked[:6]
        while len(chosen)<COUNTS['anchors_per_bank']:
            remaining=[r for r in candidates if r['index'] not in {c['index'] for c in chosen}]
            def separation(r):
                return min(float(np.sum((np.array(r['x'])-c['x'])**2)) for c in chosen)
            chosen.append(max(remaining, key=separation))
        for i,r in enumerate(chosen):
            out.append(dict(bank=bank, index=i, x=r['x'], parameters=decode(r['x']),
                            old_audit_index=r['index'], old_panel=r['panel'],
                            old_retention_pct=r['parameter_acceptance_pct'], name=r['name']+'_'+r['side']))
    return out

def cases(previous=PREVIOUS):
    out=[dict(**r) for r in old_cases()]
    for r in out:
        r['status']='historical_location_fresh_markets'
    audit=json.loads((Path(previous)/'endpoint_audit/audit_results.json').read_text())
    selected=sorted([r for r in audit if not r['empty']], key=lambda r:r['parameter_acceptance_pct'])[:3]
    for r in selected:
        out.append(dict(truth_id=len(out), name='focal_'+r['name']+'_'+r['side'], kind='focal_endpoint',
                        status='historical_location_fresh_markets', parameters=decode(r['x']),
                        original_audit_index=r['index'], original_bank=r['bank']))
    for group in ('bulk','faces','corners'):
        for j,x in enumerate(draw(3, 950500+('bulk','faces','corners').index(group), group)):
            out.append(dict(truth_id=len(out), name=f'new_v05_{group}_{j+1}', kind=group,
                            status='new_location', parameters=decode(x)))
    return out

def specification():
    return dict(version=VERSION, counts=COUNTS, anchors=anchors(), cases=cases(),
       anchor_selection='Per bank six lowest historical endpoint acceptance rates, then six farthest-point locations from the remaining historical endpoints.',
       observation_target=.965, simultaneous_tolerance_confidence=.95,
       tolerance_family_size=2*len(PANELS)*COUNTS['anchors_per_bank'],
       calibration='At each anchor use the smallest order statistic with binomial confidence at least 1-0.05/96 for population content 0.965. Per model take max(1, all 12 score-ratio tolerance limits); multiply its original cutoff everywhere.',
       inherited_response_model_and_allowance_unchanged=True,
       independent_calibration_banks=True, validation_markets_shared_across_models=True,
       main_budgets=MAIN_BUDGETS, larger_budgets=LARGE_BUDGETS,
       search_check_replicates=[0,1], differential_evolution_check='O1 first replicate at each of 36 settings in both banks; 160 generations and 96 population members; fixed seeds; local polish; expand only.',
       endpoint_audit_cases=[0,7,24,27,30,33], endpoint_audit_panels=['O1','O6','O7'],
       decision_gates=dict(
           simulation='Zero failed simulations; accounting residual below 1e-8; all exported releases reproduce their summaries.',
           calibration='Every bank/panel/case observed retention at least 0.965 in 512 fresh markets. Additionally each O6 focal case in both banks has one-sided exact 95% lower bound at least 0.965. Finite-case gate only.',
           response='Each bank/panel pooled response coverage has one-sided exact 95% lower bound at least 0.95; every 16-market cell covers at least 15 targets. Oracle uncertainty is also reported.',
           search='No empty-set reversal and no endpoint movement above 0.005 percentage point in any larger-budget or independent DE comparison.',
           endpoint_response='All 72 selected endpoint reference means and their 99% intervals lie within the inherited approximation allowances.',
           information='Report linked versus unlinked widths with paired Monte Carlo error; positive linkage gain is not a passing requirement.'),
       stop_rule='One fixed repair and one fresh validation. Do not retune after validation. Any failed gate keeps the large confirmation run on hold.',
       scope='Simultaneous finite-anchor observation tolerance statement, plus fresh finite-case diagnostics. No uniform parameter-conditional coverage claim and no certified global extrema.',
       original_confirmation_seeds_used=False)

def save_spec(root=ROOT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    content=json.dumps(specification(),indent=2)+'\n';p=root/'specification.json'
    if p.exists() and p.read_text()!=content:raise RuntimeError('Specification changed')
    p.write_text(content)
    return specification()
