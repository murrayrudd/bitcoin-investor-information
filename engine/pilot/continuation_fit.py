"""Response-directed multistart inversion and an independent global-search check.

Fits see only the portable model and released summary. No generating parameter,
future outcome, or oracle mean enters either search. Global optimality is not
claimed. Increasing budgets retains every previously found feasible endpoint.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.stats import qmc
from .repair_fit import RepairedFitter,POLY,basis
from .inference_features import vector

class DirectedFitter(RepairedFitter):
    def __init__(self,path):
        super().__init__(path)
        corners=np.array(list(itertools.product((-1.,1.),repeat=9)))
        sobol=qmc.Sobol(9,scramble=True,seed=950501).random_base2(13)*2-1
        self.pool=np.vstack([sobol,corners,self.train_x])
        f=POLY.transform(self.pool)
        self.pool_mean=f@self.wcoef;self.pool_cutoff=np.exp(f@self.cutoff_coef)
        self.pool_response=f@self.response_coef

    def _functions(self,y):
        y=np.asarray(y,float)
        if y.shape!=(self.coef.shape[1],) or not np.isfinite(y).all():raise ValueError('Invalid released summary')
        cache={};yw=y@self.white
        def evaluate(x):
            if 'x' not in cache or not np.array_equal(cache['x'],x):
                v,j=basis(x,True);res=v@self.wcoef-yw;cut=np.exp(v@self.cutoff_coef);score=res@res
                cache.update(x=np.array(x).copy(),ratio=float(score/cut),response=float(v@self.response_coef),
                  ratio_grad=(2*(j@self.wcoef)@res-score*(j@self.cutoff_coef))/cut,
                  response_grad=j@self.response_coef)
            return cache
        def score(x):e=evaluate(x);return e['ratio'],e['ratio_grad']
        def objective(x,sign):e=evaluate(x);return sign*e['response'],sign*e['response_grad']
        constraint=dict(type='ineq',fun=lambda x:1-evaluate(x)['ratio'],jac=lambda x:-evaluate(x)['ratio_grad'])
        return evaluate,score,objective,constraint

    def _starts(self,y,n):
        ratios=np.sum((self.pool_mean-np.asarray(y)@self.white)**2,axis=1)/self.pool_cutoff
        near=np.argsort(ratios)
        plausible=np.flatnonzero(ratios<=1.25)
        if len(plausible)<128:plausible=near[:1024]
        lower=plausible[np.argsort(self.pool_response[plausible])]
        upper=plausible[np.argsort(-self.pool_response[plausible])]
        selected=[]
        # Each block sends starts toward fit, both response tails, and a broad
        # parameter-space location. Diversity is within a response-directed pool.
        for i in range(n):
            mode=i%4
            candidates=(near[:512],lower[:512],upper[:512],near)[mode]
            available=np.array([j for j in candidates if int(j) not in selected],dtype=int)
            if not len(available):available=np.array([j for j in near if int(j) not in selected])
            if not selected or (i//4)%3==0:
                j=int(available[0])
            else:
                d=np.min(np.sum((self.pool[available,None,:]-self.pool[selected][None,:,:])**2,axis=2),axis=1)
                j=int(available[np.argmax(d)])
            selected.append(j)
        return [self.pool[j].copy() for j in selected]

    def fit(self,y,budgets=(24,),fixed_selection=None):
        if fixed_selection is not None:raise ValueError('Primary continuation treats participation as unknown')
        if sorted(set(budgets))!=list(budgets) or min(budgets)<1:raise ValueError('Invalid budgets')
        evaluate,score,objective,constraint=self._functions(y)
        starts=self._starts(y,max(budgets));bounds=[(-1.,1.)]*9
        lo=np.inf;hi=-np.inf;xlo=xhi=None;best=np.inf;bestx=None;rows=[]
        counts=dict(score_minimizations=0,endpoint_optimizations=0,optimizer_failures=0,feasible_endpoint_candidates=0)
        def accept(x):
            nonlocal lo,hi,xlo,xhi,best,bestx
            if not np.isfinite(x).all() or np.any(np.abs(x)>1+1e-8):return False
            e=evaluate(x)
            if e['ratio']<best:best=e['ratio'];bestx=np.array(x).copy()
            if e['ratio']>1+1e-7:return False
            counts['feasible_endpoint_candidates']+=1
            if e['response']<lo:lo=e['response'];xlo=np.array(x).copy()
            if e['response']>hi:hi=e['response'];xhi=np.array(x).copy()
            return True
        for i,start in enumerate(starts):
            accept(start);endpoint_feasible=False
            # Optimize the response directly; do not first collapse every start
            # into a discrepancy minimum that can lie in a different basin.
            for sign in (1,-1):
                opt=minimize(lambda x:objective(x,sign),start,jac=True,bounds=bounds,
                  constraints=constraint,method='SLSQP',options=dict(maxiter=240,ftol=1e-10))
                counts['endpoint_optimizations']+=1;counts['optimizer_failures']+=int(not opt.success)
                endpoint_feasible=accept(opt.x) or endpoint_feasible
            if i%4==0 or not endpoint_feasible:
                res=minimize(score,start,jac=True,bounds=bounds,method='L-BFGS-B',
                  options=dict(maxiter=240,ftol=1e-11,gtol=1e-7))
                counts['score_minimizations']+=1;counts['optimizer_failures']+=int(not res.success)
                if accept(res.x):
                    for sign in (1,-1):
                        opt=minimize(lambda x:objective(x,sign),res.x,jac=True,bounds=bounds,
                          constraints=constraint,method='SLSQP',options=dict(maxiter=240,ftol=1e-10))
                        counts['endpoint_optimizations']+=1;counts['optimizer_failures']+=int(not opt.success);accept(opt.x)
            if i+1 in budgets:
                empty=xlo is None
                rows.append(dict(starts=i+1,empty=empty,lower=None if empty else lo-self.error,
                   upper=None if empty else hi+self.error,width=None if empty else hi-lo+2*self.error,
                   lower_x=None if empty else xlo.tolist(),upper_x=None if empty else xhi.tolist(),
                   best_ratio=float(best),best_x=None if bestx is None else bestx.tolist(),
                   response_error_pp=self.error,counts=counts.copy(),search='response_directed'))
        return rows

    def global_check(self,y,initial,seed=950502,iterations=160,population=96):
        evaluate,score,objective,constraint=self._functions(y);yw=np.asarray(y)@self.white
        best=dict(initial);lo=np.inf if best['empty'] else best['lower']+self.error
        hi=-np.inf if best['empty'] else best['upper']-self.error
        xlo=None if best['empty'] else np.array(best['lower_x']);xhi=None if best['empty'] else np.array(best['upper_x'])
        def quantities(x):
            x=np.asarray(x);single=x.ndim==1;a=x[None,:] if single else x.T
            f=POLY.transform(a);r=f@self.wcoef-yw;ratios=np.sum(r*r,axis=1)/np.exp(f@self.cutoff_coef)
            response=f@self.response_coef
            return (ratios[0],response[0]) if single else (ratios,response)
        diagnostics=[]
        for si,sign in enumerate((1,-1)):
            pop=qmc.LatinHypercube(9,seed=seed+si).random(population)*2-1
            pop[:8]=self._starts(y,8)
            if not initial['empty']:
                pop[0]=initial['lower_x'];pop[1]=initial['upper_x']
            nlc=NonlinearConstraint(lambda x:np.atleast_1d(quantities(x)[0]) if np.asarray(x).ndim==1 else quantities(x)[0][None,:],-np.inf,1.)
            def obj(x):
                if np.asarray(x).ndim==2 and np.asarray(x).shape[1]==0:return np.empty(0)
                return sign*quantities(x)[1]
            opt=differential_evolution(obj,[(-1.,1.)]*9,init=pop,rng=np.random.default_rng(seed+si),
               constraints=(nlc,),maxiter=iterations,tol=1e-9,atol=1e-10,polish=False,
               vectorized=True,updating='deferred')
            ratios,responses=quantities(opt.population.T)
            order=np.argsort(sign*responses+1000*np.maximum(ratios-1,0))[:6]
            for start in [opt.x,*opt.population[order]]:
                local=minimize(lambda x:objective(x,sign),start,jac=True,bounds=[(-1.,1.)]*9,
                  constraints=constraint,method='SLSQP',options=dict(maxiter=300,ftol=1e-10))
                for x in (start,local.x):
                    e=evaluate(x)
                    if e['ratio']<=1+1e-7 and np.all(np.abs(x)<=1+1e-8):
                        if e['response']<lo:lo=e['response'];xlo=x.copy()
                        if e['response']>hi:hi=e['response'];xhi=x.copy()
            diagnostics.append(dict(sign=sign,iterations=int(opt.nit),success=bool(opt.success),message=str(opt.message)))
        empty=xlo is None
        best.update(empty=empty,lower=None if empty else lo-self.error,upper=None if empty else hi+self.error,
           width=None if empty else hi-lo+2*self.error,lower_x=None if empty else xlo.tolist(),
           upper_x=None if empty else xhi.tolist(),global_check=diagnostics)
        return best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--release',required=True)
    p.add_argument('--output',required=True);p.add_argument('--starts',type=int,default=64);a=p.parse_args()
    m=DirectedFitter(a.model)
    with np.load(a.release,allow_pickle=False) as z:release={k:z[k] for k in z.files}
    rows=m.fit(vector(release,m.meta['panel']),budgets=(a.starts,))
    Path(a.output).write_text(json.dumps(dict(model=m.meta,intervals=rows),indent=2,allow_nan=False)+'\n')
