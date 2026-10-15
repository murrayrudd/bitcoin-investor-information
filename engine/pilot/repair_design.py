"""Version 0.4 development design and independently reserved evaluation cases.

All v0.3 evaluation cases are development information in this revision.
The original confirmation seeds are not used by this module.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from .expanded_design import decode, encode, design_points, truths, PARAMETERS, BOUNDS

VERSION = '0.4.0'
ALPHA_OBS = .035
ALPHA_RESPONSE = .01
ALPHA_REFERENCE = .005
GROUPS = ('bulk', 'faces', 'corners')
COUNTS = dict(corners=512, corner_reps=8, faces=192, face_reps=12,
              endpoints=96, endpoint_reps=16, anchor_reps=32,
              check=96, check_reps=16, quantile=384, quantile_reps=24,
              calibration_per_group=599, response_per_group=99, response_reps=32,
              evaluation_reps=80, oracle_reps=512, endpoint_audit_reps=128)

def draw(n, seed, group):
    rng = np.random.default_rng(seed)
    if group == 'bulk':
        return design_points(n, seed, iid=True)[0]
    x = rng.uniform(-1, 1, (n, 9))
    if group == 'faces':
        for row in x:
            ids = rng.choice(9, int(rng.integers(1, 5)), replace=False)
            row[ids] = rng.choice([-1., 1.], len(ids))*rng.uniform(.8, 1., len(ids))
    elif group == 'corners':
        x = rng.choice([-1., 1.], (n, 9))*rng.uniform(.8, 1., (n, 9))
    else:
        raise ValueError(group)
    return x

def seed(bank, role, index, rep):
    roles = {'train':0, 'check':1, 'quantile':2, 'calibration':3, 'response':4,
             'evaluation':5, 'oracle':6, 'audit':7, 'construction':8}
    if not (bank in (0, 1) and 0 <= index < 10000 and 0 <= rep < 1000):
        raise ValueError('Seed coordinate outside allocated range')
    return 600_000_000+bank*100_000_000+roles[role]*10_000_000+index*1000+rep

def endpoint_support(previous, bank):
    values = []
    for file in sorted((Path(previous)/'fits').glob(f'bank_{bank}_*.json')):
        for row in json.loads(file.read_text()):
            if row['starts'] == 8 and row['participation'] == 'unknown' and row['replicate'] in (0, 10, 20):
                for side in ('lower_x', 'upper_x'):
                    if row[side] is not None:
                        values.append(row[side])
    x = np.unique(np.round(values, 12), axis=0)
    # Farthest-point allocation is based solely on previously reported fits.
    chosen = [int(np.argmax(np.sum(x*x, axis=1)))]
    distance = np.sum((x-x[chosen[0]])**2, axis=1)
    while len(chosen) < COUNTS['endpoints']:
        i = int(np.argmax(distance)); chosen.append(i)
        distance = np.minimum(distance, np.sum((x-x[i])**2, axis=1))
    return x[chosen]

def training_points(previous, bank):
    rng = np.random.default_rng(840000+bank)
    corners = np.array(list(itertools.product((-1., 1.), repeat=9)))
    # Shell support varies the radius to identify curvature, not just corner values.
    corners *= rng.uniform(.8, 1., corners.shape)
    items = [(x, COUNTS['corner_reps'], 'corner_support') for x in corners]
    items += [(x, COUNTS['face_reps'], 'face_support') for x in draw(COUNTS['faces'], 841000+bank, 'faces')]
    items += [(x, COUNTS['endpoint_reps'], 'old_endpoint_support') for x in endpoint_support(previous, bank)]
    items += [(encode(t['parameters']), COUNTS['anchor_reps'], 'old_case_support') for t in truths()]
    return items

def cases():
    out = [dict(**t, status='previously_seen_setting') for t in truths()]
    # New locations fixed in code before generating their markets; no selection
    # based on any v0.4 fit, coverage result, or future response.
    for group in GROUPS:
        for j, x in enumerate(draw(4, 849100+GROUPS.index(group), group)):
            out.append(dict(truth_id=len(out), name=f'fresh_{group}_{j+1}',
                            kind=group, status='new_setting', parameters=decode(x)))
    return out

def specification():
    return dict(version=VERSION, parameters=PARAMETERS, bounds=BOUNDS.tolist(), counts=COUNTS,
                alpha_observation=ALPHA_OBS, alpha_response=ALPHA_RESPONSE,
                alpha_reference=ALPHA_REFERENCE, groups=GROUPS, cases=cases(),
                candidate_mean_degrees=[3, 4], ridge=.1, cutoff_degree=2,
                cutoff_method='Smooth conditional log-score quantile regression; independent rank correction in three design components; largest correction applies everywhere.',
                response_method='Polynomial response regression excludes participation; maximum of three independently rank-calibrated error allowances.',
                claim='Finite-sample component-marginal observation calibration. Conditional coverage is an empirical diagnostic, not a proven continuous-domain guarantee.',
                seed_rule='repair_design.seed; observation seed is market seed plus 2,000,000,000.',
                original_confirmation_seeds_used=False)

def freeze_spec(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(specification(), indent=2)+'\n'
    path = root/'specification.json'
    if path.exists() and path.read_text() != text:
        raise RuntimeError('Existing specification differs; use a new output directory')
    path.write_text(text)
