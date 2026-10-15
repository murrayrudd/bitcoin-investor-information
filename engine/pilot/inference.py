"""Calibrated finite-catalog confidence sets; analyst-only fitting interface."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .inference_design import ALPHA_TEST,ALPHA_REF,SHRINKAGE,N_TRAIN,N_CAL,N_REF


def rank_pvalues(calibration_scores,observed_scores):
    c=np.asarray(calibration_scores);s=np.asarray(observed_scores)
    if c.ndim!=2 or s.shape!=(c.shape[0],):raise ValueError('Score shapes do not match')
    return (1+(c>=s[:,None]).sum(axis=1))/(c.shape[1]+1)


def pooled_precision(training,primary,shrinkage=SHRINKAGE):
    x=np.asarray(training)[primary]
    residual=(x-x.mean(axis=1,keepdims=True)).reshape(-1,x.shape[-1])
    covariance=residual.T@residual/(len(primary)*(x.shape[1]-1))
    diagonal=np.diag(covariance)
    if np.min(diagonal)<=1e-20:raise ValueError('Constant feature in training covariance')
    covariance=(1-shrinkage)*covariance+shrinkage*np.diag(diagonal)
    return np.linalg.inv(covariance)


def scores(features,means,precision):
    delta=np.asarray(features)-means
    return np.einsum('ki,ij,kj->k',delta,precision,delta)


@dataclass
class FitModel:
    panel:str
    assumption:str
    catalog:list
    means:np.ndarray
    precision:np.ndarray
    calibration:np.ndarray
    calibration_scores:np.ndarray
    response_mean:np.ndarray
    response_halfwidth:np.ndarray
    _bootstrap:dict

    @classmethod
    def load(cls,output,panel,assumption):
        out=Path(output)
        spec=json.loads((out/'analyst_catalog.json').read_text())
        catalog=spec['catalog'];train=[];cal=[];response=[]
        for row in catalog:
            with np.load(out/'candidate_banks'/f"candidate_{row['candidate_id']:03d}.npz",allow_pickle=False) as z:
                metadata=json.loads(str(z['metadata']))
                if metadata['failures']:raise ValueError('Candidate bank contains failed simulations')
                a=z[f'train_{assumption}_{panel}'];b=z[f'calibration_{assumption}_{panel}'];r=z['reference_response']
                if len(a)!=N_TRAIN or len(b)!=N_CAL or len(r)!=N_REF:raise ValueError('Incomplete bank')
                train.append(a);cal.append(b);response.append(r)
        train=np.asarray(train);cal=np.asarray(cal);response=np.asarray(response)
        means=train.mean(axis=1)
        primary=np.array([i for i,r in enumerate(catalog) if r['tier']=='primary'])
        precision=pooled_precision(train,primary)
        delta=cal-means[:,None,:]
        cscore=np.einsum('kni,ij,knj->kn',delta,precision,delta)
        halfwidth=t.ppf(1-ALPHA_REF/2,N_REF-1)*response.std(axis=1,ddof=1)/np.sqrt(N_REF)
        return cls(panel,assumption,catalog,means,precision,cal,cscore,response.mean(axis=1),halfwidth,{})

    def indices(self,grid):
        if grid=='primary':return np.array([i for i,r in enumerate(self.catalog) if r['tier']=='primary'])
        if grid=='refined':return np.arange(len(self.catalog))
        raise ValueError(grid)

    def bootstrap_interval(self,best,indices,grid):
        key=(grid,int(best))
        if key in self._bootstrap:return self._bootstrap[key]
        family=self.catalog[best]['family']
        same=indices[[self.catalog[i]['family']==family for i in indices]]
        estimates=[]
        for feature in self.calibration[best]:
            fit=scores(feature,self.means[same],self.precision)
            chosen=same[np.argmin(fit)]
            estimates.append(self.response_mean[chosen])
        lo,hi=np.quantile(estimates,[.025,.975])
        center=self.response_mean[best];mc=self.response_halfwidth[best]
        interval=(float(2*center-hi-mc),float(2*center-lo+mc))
        self._bootstrap[key]=interval
        return interval

    def fit(self,feature,grid='refined'):
        indices=self.indices(grid)
        discrepancy=scores(feature,self.means[indices],self.precision)
        p=rank_pvalues(self.calibration_scores[indices],discrepancy)
        retained=indices[p>ALPHA_TEST]
        best=indices[np.argmin(discrepancy)]
        lo=hi=None
        if len(retained):
            lo=float(np.min(self.response_mean[retained]-self.response_halfwidth[retained]))
            hi=float(np.max(self.response_mean[retained]+self.response_halfwidth[retained]))
        blo,bhi=self.bootstrap_interval(best,indices,grid)
        return dict(panel=self.panel,assumption=self.assumption,grid=grid,
          retained=retained.tolist(),retained_families=sorted(set(self.catalog[i]['family'] for i in retained)),
          lower=lo,upper=hi,width=None if lo is None else hi-lo,
          best=int(best),best_family=self.catalog[best]['family'],point=float(self.response_mean[best]),
          bootstrap_lower=blo,bootstrap_upper=bhi,bootstrap_width=bhi-blo,
          min_discrepancy=float(discrepancy.min()),maximum_pvalue=float(p.max()),
          candidate_ids=indices.tolist(),pvalues=p.tolist())
