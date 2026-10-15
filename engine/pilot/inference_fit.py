"""Fit a released dataset without reading evaluator truth or future outcomes.

Example: python -m pilot.inference_fit --bank results/inference_pilot \
    --data results/inference_pilot/evaluation/analyst/RELEASE_O6.npz \
    --assumption clean --output fitted-release.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .inference import FitModel
from .inference_features import vector


def run(bank,data,assumption,output,grid='refined'):
    with np.load(data,allow_pickle=False) as release:
        panel=str(release['panel'])
        feature=vector(release,panel)
    model=FitModel.load(bank,panel,assumption)
    fit=model.fit(feature,grid)
    fit['scope']='Finite candidate catalog under the specified model and participation assumption; no continuous-domain coverage guarantee.'
    fit['response_units']='Percentage price change; interval widths are percentage points.'
    Path(output).write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n')
    return fit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--data',required=True)
    p.add_argument('--assumption',choices=['clean','selective'],required=True)
    p.add_argument('--grid',choices=['primary','refined'],default='refined')
    p.add_argument('--output',required=True)
    a=p.parse_args();fit=run(a.bank,a.data,a.assumption,a.output,a.grid)
    print(json.dumps({k:fit[k] for k in ('panel','assumption','grid','lower','upper','best_family')},indent=2))
