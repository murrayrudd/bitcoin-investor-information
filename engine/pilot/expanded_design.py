"""Version 0.3: frozen continuous-domain expansion and untouched validation."""
from dataclasses import replace, asdict
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
from .config import Config

VERSION = '0.3.0'
PARAMETERS = ['belief_location','risk_median','commitment_median','wealth_dispersion',
              'belief_cash_correlation','belief_drift','funding_rate','allowance_growth','log_selection_odds']
BOUNDS = np.array([[-.3,.3],[1.2,3.4],[.35,.75],[.55,1.15],[-.2,.8],
                   [-.03,.03],[0,.01],[-.02,.02],[np.log(.25),np.log(6.)]])
COUNTS = dict(train_points=1024,train_reps=8,check_points=128,check_reps=12,
              calibration_points=1999,response_points=199,response_reps=40,
              evaluation_reps=40,oracle_reps=640)
ALPHA_OBS=.035
ALPHA_EMULATOR=.01
ALPHA_REFERENCE=.005
SHRINKAGE=.10
DEGREE=3
RIDGE=.1
SEARCH_BUDGETS=(2,8,24)
SEARCH_TOLERANCE=.005  # response percentage points; diagnostic, not a guarantee

def decode(x):
    values=BOUNDS[:,0]+(np.asarray(x)+1)/2*(BOUNDS[:,1]-BOUNDS[:,0])
    pars={k:float(v) for k,v in zip(PARAMETERS[:-1],values[:-1])}
    pars['selection_odds_ratio']=float(np.exp(values[-1]))
    return pars

def encode(parameters):
    cfg=replace(Config(),**parameters)
    values=np.array([getattr(cfg,k) for k in PARAMETERS[:-1]]+[np.log(cfg.selection_odds_ratio)])
    return 2*(values-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])-1

def design_points(n,seed,iid=False):
    """Calibration iid design: 3/4 mixed; 1/12 each of three pure subspaces.

    Coordinates uniform in the declared box (log-uniform selection odds).
    Training uses Latin hypercube coordinates with the same independent labels.
    This design is a simulation allocation, not an empirical prior.
    """
    rng=np.random.default_rng(seed)
    x=(rng.uniform(size=(n,9)) if iid else qmc.LatinHypercube(9,rng=rng).random(n))*2-1
    strata=rng.choice(4,n,p=[.75,1/12,1/12,1/12])
    zero=encode({'selection_odds_ratio':1})
    for i in range(n):
        if strata[i]:
            active=4+int(strata[i])
            for j in (5,6,7):
                if j!=active:x[i,j]=zero[j]
    return x,strata

def truths():
    specs=[
      ('pure_belief','pure',dict(belief_location=.055,risk_median=2.15,commitment_median=.59,wealth_dispersion=.96,belief_cash_correlation=.48,belief_drift=.018,selection_odds_ratio=1)),
      ('pure_funding','pure',dict(belief_location=-.06,risk_median=1.85,commitment_median=.51,wealth_dispersion=.84,belief_cash_correlation=.64,funding_rate=.0075,selection_odds_ratio=2)),
      ('pure_limits','pure',dict(belief_location=.025,risk_median=2.2,commitment_median=.6,wealth_dispersion=.8,belief_cash_correlation=.5,allowance_growth=.015,selection_odds_ratio=4)),
      ('mixed_previous_challenge','mixed',dict(belief_drift=.01,funding_rate=.005,allowance_growth=.01,selection_odds_ratio=1)),
      ('mixed_reinforcing','mixed',dict(belief_location=.1,risk_median=2.7,commitment_median=.65,wealth_dispersion=1.05,belief_cash_correlation=.3,belief_drift=.024,funding_rate=.007,allowance_growth=.014,selection_odds_ratio=3)),
      ('mixed_offsetting','mixed',dict(belief_location=-.1,risk_median=1.6,commitment_median=.48,wealth_dispersion=.7,belief_cash_correlation=.7,belief_drift=-.02,funding_rate=.008,allowance_growth=.012,selection_odds_ratio=1.5)),
      ('mixed_tightening','mixed',dict(belief_location=.2,risk_median=2.3,commitment_median=.6,wealth_dispersion=1.,belief_cash_correlation=-.05,belief_drift=.016,funding_rate=.006,allowance_growth=-.015,selection_odds_ratio=4)),
      ('mixed_declining','mixed',dict(belief_location=-.22,risk_median=3.,commitment_median=.68,wealth_dispersion=.65,belief_cash_correlation=.2,belief_drift=-.018,funding_rate=.002,allowance_growth=-.012,selection_odds_ratio=.5)),
      ('mixed_reverse_selection','mixed',dict(belief_location=.12,risk_median=1.9,commitment_median=.45,wealth_dispersion=1.1,belief_cash_correlation=.72,belief_drift=.008,funding_rate=.004,allowance_growth=.007,selection_odds_ratio=.3)),
      ('boundary_high_funding','boundary',dict(belief_location=.28,risk_median=3.3,commitment_median=.73,wealth_dispersion=1.12,belief_cash_correlation=.77,belief_drift=.028,funding_rate=.0098,allowance_growth=.019,selection_odds_ratio=5.7)),
      ('boundary_low_resources','boundary',dict(belief_location=-.28,risk_median=1.3,commitment_median=.73,wealth_dispersion=.58,belief_cash_correlation=-.17,belief_drift=-.028,funding_rate=.0003,allowance_growth=-.019,selection_odds_ratio=.27)),
      ('boundary_opposed_drifts','boundary',dict(belief_location=.26,risk_median=1.3,commitment_median=.37,wealth_dispersion=1.1,belief_cash_correlation=-.15,belief_drift=.027,funding_rate=.009,allowance_growth=-.018,selection_odds_ratio=5.5)),
    ]
    return [dict(truth_id=i,name=n,kind=k,parameters=p) for i,(n,k,p) in enumerate(specs)]

def seed(bank,role,index,rep):
    if role in ('train','check','calibration','response'):
        return 200_000_000+bank*50_000_000+{'train':0,'check':3_000_000,'calibration':5_000_000,'response':10_000_000}[role]+index*1000+rep
    return {'evaluation':400_000_000,'oracle':410_000_000,'audit':420_000_000}[role]+index*1000+rep

def freeze(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=dict(version=VERSION,config=asdict(Config()),parameters=PARAMETERS,bounds=BOUNDS.tolist(),counts=COUNTS,
              alpha_observation=ALPHA_OBS,alpha_response_emulator=ALPHA_EMULATOR,alpha_reference=ALPHA_REFERENCE,
              degree=DEGREE,ridge=RIDGE,shrinkage=SHRINKAGE,search_budgets=SEARCH_BUDGETS,search_tolerance_pp=SEARCH_TOLERANCE,
              calibration_scope='Design-marginal over the stated iid parameter mixture; NOT pointwise continuous-domain 95% coverage.',
              design='3/4 uniform mixed box; 1/12 each pure belief, funding, allowance subspace; log-uniform participation odds 0.25 to 6.',
              response='Expected month-19 percent price response to a 5% aggregate-cash transfer to a fixed random quarter of participants.',
              oracle='Independent mean over 640 markets at each validation setting; never supplied to fitting.',
              seed_rule='expanded_design.seed; observation seed = market seed + 2,000,000,000; old confirmation seeds remain untouched.',
              truth_cases=truths())
    text=json.dumps(spec,indent=2,allow_nan=False)+'\n';path=out/'specification.json'
    if path.exists() and path.read_text()!=text:raise RuntimeError('Frozen specification differs')
    path.write_text(text)
    return spec
