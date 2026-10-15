"""Replay archived data to test the confirmation driver before new draws."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import confirmation_runner as r
from confirmation_summary import Bootstrap

cases=r.design()
assert [sum(c['component']==g for c in cases) for g in r.COMPONENTS]==list(r.NSET)
assert sum(c['endpoint_audit'] for c in cases)==12
assert all(np.max(np.abs(c['x']))<=1 for c in cases)
for c in cases:
    if c['component'] in ('belief','funding','allowance'):
        active={'belief':'belief_drift','funding':'funding_rate','allowance':'allowance_growth'}[c['component']]
        assert all(abs(c['parameters'][p])<1e-15 for p in ('belief_drift','funding_rate','allowance_growth') if p!=active)
b=Bootstrap(cases)
for g in b.weights:
    np.testing.assert_allclose(b.draws[g].sum(axis=1),1.)
    np.testing.assert_allclose(b.weights[g].sum(),1.)
np.testing.assert_allclose(b.estimate('primary',np.ones(256))['estimate'],1.)
r.enable_compiled(required=True)
with np.load(r.PARENT/'validation/case_00.npz',allow_pickle=False) as z:
    old={k:z[k] for k in z.files}
meta=json.loads(str(old['metadata']))
with tempfile.TemporaryDirectory(prefix='confirmation_replay_') as td:
    r.ROOT=Path(td)
    r.simulation_job(dict(path=str(r.ROOT/'replay.npz'),role='evaluation',case_id=0,
        parameters=meta['parameters'],seeds=old['seeds'][:4].tolist()))
    with np.load(r.ROOT/'replay.npz',allow_pickle=False) as z:
        for k in ('response','O1','O2','O6','O7'): np.testing.assert_array_equal(z[k],old[k][:4])
    (r.ROOT/'models').mkdir()
    for p in (r.PARENT/'models').glob('*.npz'): shutil.copy2(p,r.ROOT/'models'/p.name)
    r.fit_init()
    original_seed=r.seed
    r.seed=lambda role,index,rep=0: int(old['seeds'][rep]) if role=='evaluation' else original_seed(role,index,rep)
    r.fit_one(0,'O2',0,0)
    rows=r.read_json(r.fit_path(0,'O2',0,0));previous=r.read_json(r.PARENT/'fits/bank_0_O2_case_00.json')
    for row in rows:
        if row['method']=='primary':
            prev=next(x for x in previous if x['replicate']==row['replicate'] and x['starts']==24 and x['method']=='v0.5')
            for key in ('empty','lower','upper','width','lower_x','upper_x','best_ratio','counts'):
                assert row[key]==prev[key],key
print('PASS: design allocations, drift restrictions, bounds, audits, bootstrap weights, and exact archived simulation/feature/fit replay.')
