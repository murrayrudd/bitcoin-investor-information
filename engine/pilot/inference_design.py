"""Frozen catalog, challenge cases, and independent simulation seed banks."""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .config import Config,ENVIRONMENTS

VERSION='0.2.0'
N_TRAIN=32
N_CAL=99
N_REF=64
N_EVAL=50
N_ORACLE=768
ALPHA_TEST=.04
ALPHA_REF=.01
SHRINKAGE=.10
WIDTH_THRESHOLD=.10
PARAMETERS=['belief_location','risk_median','commitment_median','wealth_dispersion','belief_cash_correlation']
BASE_BOUNDS=[[-.3,.3],[1.2,3.4],[.35,.75],[.55,1.15],[-.2,.8]]
DRIFTS={'belief':('belief_drift',[-.03,.03]),'funding':('funding_rate',[0,.01]),'mandate':('allowance_growth',[-.02,.02])}


def candidate_catalog():
    rows=[]
    for fi,family in enumerate(ENVIRONMENTS):
        drift,bounds=DRIFTS[family]
        names=PARAMETERS+[drift]
        limits=np.asarray(BASE_BOUNDS+[bounds],float)
        base=asdict(replace(Config(),**ENVIRONMENTS[family]))
        anchor=np.array([base[k] for k in names])
        step=np.array([.10,.30,.08,.12,.20,{'belief':.006,'funding':.002,'mandate':.006}[family]])
        def add(values,tier,kind):
            changes={k:float(v) for k,v in zip(names,values)}
            cfg=replace(Config(),**changes).validate()
            # At upper domain boundaries the clipped positive axial point is
            # the anchor; de-duplicate rather than count it as a new candidate.
            if any(r['family']==family and r['parameters']==changes for r in rows):return
            rows.append(dict(candidate_id=len(rows),family=family,tier=tier,kind=kind,parameters=changes))
        add(anchor,'primary','anchor')
        for d in range(6):
            for sign in (-1,1):
                x=anchor.copy();x[d]=np.clip(x[d]+sign*step[d],*limits[d])
                add(x,'primary','axial')
        lhs=qmc.LatinHypercube(d=6,rng=np.random.default_rng(82001+fi)).random(16)
        for x in qmc.scale(lhs,limits[:,0],limits[:,1]):add(x,'primary','global_lhs')
        local_low=np.maximum(limits[:,0],anchor-step)
        local_high=np.minimum(limits[:,1],anchor+step)
        lhs=qmc.LatinHypercube(d=6,rng=np.random.default_rng(83001+fi)).random(32)
        for x in qmc.scale(lhs,local_low,local_high):add(x,'refined','local_lhs')
    return rows


def truth_cases():
    rows=[dict(truth_id=i,name=f'{f}_anchor',family=f,kind='on_catalog',parameters=c) for i,(f,c) in enumerate(ENVIRONMENTS.items())]
    changes=[('belief',dict(belief_location=.04,risk_median=2.12,commitment_median=.58,wealth_dispersion=.94,belief_cash_correlation=.52,belief_drift=.0185)),
      ('funding',dict(belief_location=-.04,risk_median=1.90,commitment_median=.52,wealth_dispersion=.86,belief_cash_correlation=.67,funding_rate=.009)),
      ('mandate',dict(belief_location=.035,risk_median=2.10,commitment_median=.57,wealth_dispersion=.87,belief_cash_correlation=.55,allowance_growth=.018))]
    for f,c in changes:rows.append(dict(truth_id=len(rows),name=f'{f}_between',family=f,kind='off_catalog',parameters=c))
    rows.append(dict(truth_id=len(rows),name='mixed_drift',family='mixed',kind='omitted_family',parameters=dict(belief_drift=.01,funding_rate=.005,allowance_growth=.01)))
    return rows


def seed_bank(role,index,count):
    if role in ('train','calibration','reference'):
        base=10_000_000+index*10_000+{'train':0,'calibration':1000,'reference':2000}[role]
    elif role=='evaluation':base=50_000_000+index*10_000
    elif role=='oracle':base=60_000_000+index*10_000
    else:raise ValueError(role)
    return np.arange(base,base+count,dtype=np.int64)


def freeze(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    specification=dict(version=VERSION,config=asdict(Config()),catalog=candidate_catalog(),truth_cases=truth_cases(),
      counts=dict(train=N_TRAIN,calibration=N_CAL,reference=N_REF,evaluation=N_EVAL,oracle=N_ORACLE),
      alpha_test=ALPHA_TEST,alpha_reference=ALPHA_REF,covariance_shrinkage=SHRINKAGE,
      misleading_precision_width_pp=WIDTH_THRESHOLD,seed_rule='See seed_bank; observation seed = market seed + 1,000,000,000',
      coverage_scope='Pointwise on finite catalog under the stated observation generator; no continuous-domain coverage claim.',
      covariance='Pooled within-candidate training residual covariance, separately by panel and assumed participation; primary catalog only; fixed for refinement.',
      bootstrap='Fit a single candidate, fix its family, bootstrap by refitting its 99 calibration releases within that family; basic 95% interval, expanded by selected-candidate 99% response Monte Carlo half-width.',
      error_handling='Record failures, never replace seeds; stop inferential reporting if any candidate bank is incomplete.')
    path=out/'inference_specification.json'
    content=json.dumps(specification,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=content:raise RuntimeError('Frozen specification differs; use new output directory')
    path.write_text(content)
    analyst={k:specification[k] for k in ('version','config','catalog','counts')}
    (out/'analyst_catalog.json').write_text(json.dumps(analyst,indent=2,allow_nan=False)+'\n')
    return specification
