"""Checkpointed fresh simulation banks for the inference repair."""
import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .expanded_bank import generate, init
from .expanded_design import decode
from .repair_design import COUNTS, GROUPS, draw, seed, training_points, cases, freeze_spec

def jobs(root, previous, which, banks=(0, 1)):
    root = Path(root)
    if which in ('development', 'all'):
        for bank in banks:
            for i, (x, reps, group) in enumerate(training_points(previous, bank)):
                yield make_job(root, bank, 'train', i, x, reps, group)
            for role, n, reps in [('check', COUNTS['check'], COUNTS['check_reps']),
                                  ('quantile', COUNTS['quantile'], COUNTS['quantile_reps'])]:
                for gi, group in enumerate(GROUPS):
                    for j, x in enumerate(draw(n//3, 842000+bank*100+gi*10+(role=='quantile'), group)):
                        yield make_job(root, bank, role, gi*(n//3)+j, x, reps, group)
            for role, n, reps in [('calibration', COUNTS['calibration_per_group'], 1),
                                  ('response', COUNTS['response_per_group'], COUNTS['response_reps'])]:
                for gi, group in enumerate(GROUPS):
                    for j, x in enumerate(draw(n, 843000+bank*100+gi*10+(role=='response'), group)):
                        yield make_job(root, bank, role, gi*n+j, x, reps, group)
    if which in ('validation', 'all'):
        freeze = root/'source_and_model_freeze.json'
        if not freeze.exists():
            raise RuntimeError('Freeze fitted models and inferential source before validation')
        for case in cases():
            for role, reps in [('evaluation', COUNTS['evaluation_reps']), ('oracle', COUNTS['oracle_reps'])]:
                folder = root/role; folder.mkdir(exist_ok=True)
                i = case['truth_id']
                yield dict(path=str(folder/f'truth_{i:02d}.npz'), parameters=case['parameters'], role=role,
                           seeds=[seed(0, role, i, r) for r in range(reps)])

def make_job(root, bank, role, i, x, reps, group):
    folder = root/f'bank_{bank}'/role; folder.mkdir(parents=True, exist_ok=True)
    return dict(path=str(folder/f'{i:04d}.npz'), parameters=decode(x), x=x.tolist(),
                stratum=GROUPS.index(group) if group in GROUPS else -1,
                role=role, seeds=[seed(bank, role, i, r) for r in range(reps)])

def run(root, previous, which='development', workers=8, banks=(0, 1)):
    freeze_spec(root); root=Path(root); pending=[]; existing=0
    for job in jobs(root, previous, which, banks):
        path = Path(job['path'])
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']))
                if meta['failures'] or meta['successful'] != len(job['seeds']) or not np.array_equal(z['seeds'], job['seeds']):
                    raise RuntimeError(f'Incomplete or inconsistent checkpoint {path}')
            existing += 1
        else:
            pending.append(job)
    init(); start=time.perf_counter(); completed=[]
    print(json.dumps(dict(stage=which, jobs=len(pending), existing=existing,
                          markets=sum(len(j['seeds']) for j in pending))), flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        futures=[pool.submit(generate, j) for j in pending]
        for i, future in enumerate(as_completed(futures)):
            result=future.result(); completed.append(result)
            if result['failures']:
                raise RuntimeError(result)
            if i % 25 == 0 or i+1 == len(futures):
                progress=dict(stage=which, done=i+1, total=len(futures),
                              markets=sum(r['markets'] for r in completed),
                              seconds=time.perf_counter()-start, last=result)
                print(json.dumps(progress), flush=True)
                (root/f'progress_{which}.json').write_text(json.dumps(progress, indent=2)+'\n')
    (root/f'jobs_{which}.json').write_text(json.dumps(completed, indent=2)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--previous', default='results/expanded_pilot')
    p.add_argument('--which', choices=['development', 'validation'], default='development')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--bank', type=int, choices=[0, 1])
    a=p.parse_args(); run(a.root, a.previous, a.which, a.workers, (a.bank,) if a.bank is not None else (0, 1))
