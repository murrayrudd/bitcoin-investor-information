"""Continuous projected response intervals from released summaries only.

Calibration is marginal over the frozen simulation design, not a guarantee
at every fixed parameter. Numerical endpoint optimization is approximate.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc,t
from sklearn.preprocessing import PolynomialFeatures
from .observations import PANELS
from .inference_features import vector
from .expanded_design import DEGREE,RIDGE,SHRINKAGE,ALPHA_OBS,ALPHA_EMULATOR,ALPHA_REFERENCE,COUNTS

POLY=PolynomialFeatures(DEGREE).fit(np.zeros((1,9)))
POWERS=POLY.powers_
INDICES=np.full((len(POWERS),3),9,dtype=int)
for j,p in enumerate(POWERS):
    ids=np.repeat(np.arange(9),p);INDICES[j,:len(ids)]=ids
DERIV_INDICES=np.tile(INDICES[None,:,:],(9,1,1))
for d in range(9):
    for j,p in enumerate(POWERS):
        if p[d]:DERIV_INDICES[d,j,np.flatnonzero(INDICES[j]==d)[0]]=9

def basis(x,derivative=False):
    xx=np.r_[x,1.]
    v=np.prod(xx[INDICES],axis=1)
    if not derivative:return v
    grad=np.prod(xx[DERIV_INDICES],axis=2)*POWERS.T
    return v,grad

def rank_threshold(values,alpha):
    values=np.asarray(values);k=int(math.ceil((len(values)+1)*(1-alpha)))
    if k>len(values):return float('inf')
    return float(np.sort(values)[k-1])

def ridge_fit(x,y,mask=None):
    features=POLY.transform(x)
    if mask is None:mask=np.ones(features.shape[1],dtype=bool)
    a=features[:,mask];penalty=RIDGE*np.eye(a.shape[1]);penalty[0,0]=0
    coef=np.linalg.solve(a.T@a+penalty,a.T@y)
    out=np.zeros((features.shape[1],)+y.shape[1:]);out[mask]=coef
    return out

def load_bank(folder):
    result={k:[] for k in ['x','response','stratum',*PANELS]};names=None
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['failures']:raise RuntimeError(f'Incomplete {path}')
            for k in result:
                if k in z:result[k].append(z[k])
            if meta['feature_names']:names=meta['feature_names']
    return {k:np.asarray(v) for k,v in result.items() if v},names

def train_bank(root,bank):
    root=Path(root);folder=root/f'bank_{bank}'
    train,names=load_bank(folder/'train');check,_=load_bank(folder/'check')
    cal,_=load_bank(folder/'calibration');ref,_=load_bank(folder/'response')
    expected={'train':COUNTS['train_points'],'check':COUNTS['check_points'],'calibration':COUNTS['calibration_points'],'response':COUNTS['response_points']}
    for label,data in [('train',train),('check',check),('calibration',cal),('response',ref)]:
        if len(data['x'])!=expected[label]:raise RuntimeError(f'Incomplete {label} bank')
    response_coef=ridge_fit(train['x'],train['response'].mean(axis=1),POWERS[:,8]==0)
    ref_pred=POLY.transform(ref['x'])@response_coef
    ref_mean=ref['response'].mean(axis=1)
    half=t.ppf(1-ALPHA_REFERENCE/2,ref['response'].shape[1]-1)*ref['response'].std(axis=1,ddof=1)/np.sqrt(ref['response'].shape[1])
    err=np.abs(ref_mean-ref_pred)+half
    response_error=rank_threshold(err,ALPHA_EMULATOR)
    dest=root/'models';dest.mkdir(exist_ok=True)
    check_pred=POLY.transform(check['x'])@response_coef
    diagnostics=dict(bank=bank,response_error_pp=response_error,
      response_check_rmse_pp=float(np.sqrt(np.mean((check_pred-check['response'].mean(axis=1))**2))),
      response_check_mean_mcse_pp=float(np.mean(check['response'].std(axis=1,ddof=1)/np.sqrt(check['response'].shape[1]))),
      response_reference_max_error_pp=float(err.max()),panels={})
    for panel in PANELS:
        means=train[panel].mean(axis=1);coef=ridge_fit(train['x'],means)
        for j,name in enumerate(names[panel]):
            if name.startswith('signal_'):coef[:,j]=0.
        residual=train[panel]-means[:,None,:]
        flat=residual.reshape(-1,means.shape[-1])
        covariance=flat.T@flat/(len(means)*(train[panel].shape[1]-1))
        covariance=(1-SHRINKAGE)*covariance+SHRINKAGE*np.diag(np.diag(covariance))
        precision=np.linalg.inv(covariance);white=np.linalg.cholesky(precision)
        residual_cal=cal[panel][:,0,:]-POLY.transform(cal['x'])@coef
        scores=np.sum((residual_cal@white)**2,axis=1)
        cutoff=rank_threshold(scores,ALPHA_OBS)
        residual_check=check[panel]- (POLY.transform(check['x'])@coef)[:,None,:]
        check_scores=np.sum((residual_check@white)**2,axis=2)
        mu_error=(check[panel].mean(axis=1)-POLY.transform(check['x'])@coef)/np.sqrt(np.diag(covariance))
        diagnostic=dict(cutoff=cutoff,check_true_parameter_acceptance=float((check_scores<=cutoff).mean()),
                        check_mean_feature_error_rms_sd=float(np.sqrt(np.mean(mu_error**2))),
                        check_feature_error_rms_sd={k:float(v) for k,v in zip(names[panel],np.sqrt(np.mean(mu_error**2,axis=0)))})
        diagnostics['panels'][panel]=diagnostic
        meta=dict(version='0.3.0',bank=bank,panel=panel,feature_names=names[panel],cutoff=cutoff,response_error_pp=response_error,
                  coverage_scope='Design-marginal calibration; no pointwise continuous-domain coverage guarantee; approximate endpoint search.',
                  diagnostic=diagnostic)
        np.savez_compressed(dest/f'bank_{bank}_{panel}.npz',coef=coef,white=white,response_coef=response_coef,
                            metadata=np.array(json.dumps(meta)),train_x=train['x'])
    (dest/f'bank_{bank}_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    return diagnostics

class ContinuousFitter:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.coef=z['coef'];self.white=z['white'];self.response_coef=z['response_coef'];self.meta=json.loads(str(z['metadata']))
            self.train_x=z['train_x']
        self.cutoff=self.meta['cutoff'];self.error=self.meta['response_error_pp'];self.wcoef=self.coef@self.white
        sobol=qmc.Sobol(9,scramble=True,seed=992001).random_base2(9)*2-1
        self.pool=np.vstack([sobol,self.train_x])
        self.pool_mean=POLY.transform(self.pool)@self.wcoef

    def fit(self,y,budgets=(2,8),fixed_selection=None):
        y=np.asarray(y,float)
        if len(y)!=self.coef.shape[1] or not np.isfinite(y).all():raise ValueError('Invalid released summary vector')
        yw=y@self.white;limit=max(budgets);bounds=[(-1.,1.)]*9
        pool=self.pool
        if fixed_selection is None:pool_mean=self.pool_mean
        else:
            if not -1<=fixed_selection<=1:raise ValueError('Participation outside declared domain')
            bounds[-1]=(fixed_selection,fixed_selection)
            pool=pool.copy();pool[:,-1]=fixed_selection;pool_mean=POLY.transform(pool)@self.wcoef
        scores=np.sum((pool_mean-yw)**2,axis=1);order=np.argsort(scores)
        # Alternate near-data starts and parameter-space separated starts.
        starts=[];chosen=[]
        for i in range(limit):
            if i%2==0:idx=next(int(j) for j in order if int(j) not in chosen)
            else:
                candidates=order[:256]
                distance=np.min(np.sum((pool[candidates,None,:]-pool[chosen][None,:,:])**2,axis=2),axis=1)
                idx=int(candidates[np.argmax(distance)])
            chosen.append(idx);starts.append(pool[idx])
        cache={};counts=dict(score_minimizations=0,endpoint_optimizations=0,optimizer_failures=0,feasible_endpoint_candidates=0)
        def evaluate(x):
            if 'x' not in cache or not np.array_equal(x,cache['x']):
                v,j=basis(x,True);res=v@self.wcoef-yw
                cache.update(x=x.copy(),score=float(res@res),score_grad=2*(j@self.wcoef)@res,
                             response=float(v@self.response_coef),response_grad=j@self.response_coef)
            return cache
        def score(x):e=evaluate(x);return e['score']/self.cutoff,e['score_grad']/self.cutoff
        def objective(x,sign):e=evaluate(x);return sign*e['response'],sign*e['response_grad']
        constraint={'type':'ineq','fun':lambda x:1-evaluate(x)['score']/self.cutoff,
                    'jac':lambda x:-evaluate(x)['score_grad']/self.cutoff}
        lo=float('inf');hi=-float('inf');xlo=xhi=None;best_score=float('inf');best_x=None;results=[]
        for i,start in enumerate(starts):
            res=minimize(score,start,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-11,'gtol':1e-7})
            counts['score_minimizations']+=1;counts['optimizer_failures']+=not res.success
            val=evaluate(res.x)['score']
            if val<best_score:best_score=val;best_x=res.x.copy()
            if val<=self.cutoff*(1+1e-8):
                for sign in (1,-1):
                    opt=minimize(lambda x:objective(x,sign),res.x,jac=True,bounds=bounds,constraints=constraint,
                                 method='SLSQP',options={'maxiter':180,'ftol':1e-9})
                    counts['endpoint_optimizations']+=1;counts['optimizer_failures']+=not opt.success
                    for xx in (res.x,opt.x):
                        e=evaluate(xx)
                        if e['score']<=self.cutoff*(1+1e-7):
                            counts['feasible_endpoint_candidates']+=1
                            if e['response']<lo:lo=e['response'];xlo=xx.copy()
                            if e['response']>hi:hi=e['response'];xhi=xx.copy()
            if i+1 in budgets:
                empty=xlo is None
                results.append(dict(starts=i+1,empty=empty,lower=None if empty else lo-self.error,
                  upper=None if empty else hi+self.error,width=None if empty else hi-lo+2*self.error,
                  response_error_pp=self.error,best_score=best_score,cutoff=self.cutoff,best_x=best_x.tolist(),
                  lower_x=None if empty else xlo.tolist(),upper_x=None if empty else xhi.tolist(),counts=counts.copy()))
        return results

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--root',required=True)
    fit=sub.add_parser('fit');fit.add_argument('--model',required=True);fit.add_argument('--release',required=True);fit.add_argument('--output',required=True);fit.add_argument('--starts',type=int,default=24)
    a=p.parse_args()
    if a.command=='train':
        for b in range(2):print(json.dumps(train_bank(a.root,b)),flush=True)
    else:
        model=ContinuousFitter(a.model)
        with np.load(a.release,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        result=model.fit(vector(data,model.meta['panel']),budgets=(a.starts,))
        Path(a.output).write_text(json.dumps(dict(model=model.meta,intervals=result),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
