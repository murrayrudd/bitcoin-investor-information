import unittest
import csv
import contextlib
import io
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from pilot.config import Config
from pilot.economy import (population, demand, clearing, simulate,
    discover_roots, EconomicInfeasibility)
from pilot.observations import release_panels, risk_task_ce
from pilot.features import summaries


class Verification(unittest.TestCase):
    def test_individual_choices_against_independent_utility_optimization(self):
        rng=np.random.default_rng(99143)
        maximum=0
        for j in range(200):
            c,h=rng.uniform(.02,2,2)
            floor=rng.uniform(0,.9)*c
            cap=h+rng.uniform(0,3)*c
            gamma=1.0 if j%10==0 else rng.uniform(.5,6)
            q=rng.uniform(.05,.95)
            p=rng.uniform(.25,2.1)
            upper=min(cap,h+(c-floor)/p)
            def utility(z):
                # Direct two-state utility; no reuse of the analytic FOC.
                w=(c+p*(h-z*upper)+np.array([.6,1.8])*z*upper)/(c+p*h)
                u=np.log(w) if gamma==1 else np.expm1((1-gamma)*np.log(w))/(1-gamma)
                return float(np.dot([1-q,q],u))
            opt=minimize_scalar(lambda z:-utility(z),bounds=(0,1),method="bounded",options={"xatol":1e-12})
            self.assertTrue(opt.success)
            ref=max([0.0,1.0,opt.x],key=utility)*upper
            got=demand(p,*[np.array([v]) for v in [c,h,floor,cap,q,gamma]])[0]
            maximum=max(maximum,abs(got-ref))
            self.assertAlmostEqual(got,ref,delta=3e-7*max(1,upper))

    def test_aggregate_price_against_closed_form(self):
        cfg=Config(n=100,sample_size=25)
        pop=population(cfg,np.random.default_rng(73))
        pop.floor[:]=0; pop.gamma[:]=2
        q=np.full(100,.6); cap=np.full(100,np.inf)
        a=.6/(100+1.8*100)**2
        b=.4/(100+.6*100)**2
        expected=(a*1.8+b*.6)/(a+b)
        p,c,h,d=clearing(pop.cash0,pop.holdings0,pop,q,cap,1,cfg)
        self.assertAlmostEqual(p,expected,places=10)
        self.assertAlmostEqual(c.sum(),100,places=8)
        self.assertAlmostEqual(h.sum(),100,places=8)

    def test_scale_invariance(self):
        cfg=Config(n=100,sample_size=25)
        pop=population(cfg,np.random.default_rng(74))
        q=np.full(100,.65); cap=pop.holdings0+pop.allowance0
        x=demand(1,pop.cash0,pop.holdings0,pop.floor,cap,q,pop.gamma)
        y=demand(1,pop.cash0*7,pop.holdings0*7,pop.floor*7,cap*7,q,pop.gamma)
        np.testing.assert_allclose(x*7,y,rtol=1e-12,atol=1e-12)

    def test_roots_multiple_and_zero_interval(self):
        p,roots,spans=discover_roots(lambda x:(x-.4)*(x-.8)*(x-1.4),np.linspace(.1,1.8,42),.85,1e-10)
        np.testing.assert_allclose(roots,[.4,.8,1.4],atol=1e-9)
        self.assertAlmostEqual(p,.8,places=9)
        p,roots,spans=discover_roots(lambda x: max(.7-x,0)-max(x-1.1,0),np.linspace(.1,1.8,42),.9,1e-10)
        self.assertAlmostEqual(p,.9)
        self.assertTrue(spans)

    def test_infeasible_floor_not_repaired(self):
        with self.assertRaises(EconomicInfeasibility):
            demand(1,*[np.array([v]) for v in [.1,0,.2,1,.5,2]])

    def test_quadratic_appendix_benchmark(self):
        worlds=[(2,1,0,10),(3,1,2,10),(3,1,0,1),(3,2,0,10)]
        expected_cash=[1,(1+np.sqrt(1.8))/2,1,1]
        expected_cap=[1,1,1.2,1]
        for i,(v,d,l,k) in enumerate(worlds):
            for intervention in ["baseline","cash","cap"]:
                c=np.array([2.2 if intervention=="cash" else 2.,2.])
                h=np.ones(2); floor=np.array([l,0.])
                ceiling=np.array([k+.2 if intervention=="cap" else k,10.])
                def holdings(p):
                    return np.clip((np.array([v,2])-p)/np.array([d,1]),0,
                                   np.minimum(ceiling,h+(c-floor)/p))
                p=brentq(lambda p:holdings(p).sum()-2,.1,4,xtol=1e-13)
                expected=1 if intervention=="baseline" else (expected_cash if intervention=="cash" else expected_cap)[i]
                self.assertAlmostEqual(p,expected,places=10)
                self.assertAlmostEqual((c+p*(h-holdings(p))).sum(),c.sum(),places=10)

    def test_zero_transfer_pairing_and_reproducibility(self):
        cfg=Config(n=100,sample_size=25,transfer_share=0)
        a=simulate(cfg,681); b=simulate(cfg,681)
        np.testing.assert_array_equal(a.prices,b.prices)
        self.assertAlmostEqual(a.response,0,places=12)
        self.assertAlmostEqual(a.response_six,0,places=12)
        np.testing.assert_array_equal(a.control_final.cash,a.treatment_final.cash)

    def test_cash_stock_and_funding_ledgers(self):
        cfg=Config(n=120,sample_size=30,funding_rate=.01)
        m=simulate(cfg,612)
        self.assertAlmostEqual(m.control_final.cash.sum(),124,places=7)
        self.assertAlmostEqual(m.treatment_final.cash.sum(),124+m.transfer,places=7)
        self.assertAlmostEqual(m.control_final.holdings.sum(),100,places=7)
        self.assertTrue(all(abs(r["cash_ledger_residual"])<1e-7 for r in m.ledger))

    def test_observations_cutoff_linkage_and_truth_separation(self):
        cfg=Config(n=120,sample_size=25)
        m=simulate(cfg,847)
        panels,_=release_panels(m,cfg,310,"clean")
        self.assertEqual(len(panels["O1"]["prices"]),18)
        self.assertNotIn("cash",panels["O1"])
        self.assertNotIn("gamma",panels["O6"])
        self.assertNotIn("response",panels["O6"])
        for k in ["holdings","belief","cash","commitment","cap","risk_ce"]:
            np.testing.assert_array_equal(np.sort(panels["O6"][k],axis=1),np.sort(panels["O7"][k],axis=1))
        np.testing.assert_array_equal(np.sort(panels["O6"]["cash"]-panels["O6"]["commitment"],axis=1),
                                      np.sort(panels["O7"]["cash"]-panels["O7"]["commitment"],axis=1))
        f=summaries(panels["O7"],"O7")
        self.assertFalse(any("correlation" in k for k in f))
        with self.assertRaises(ValueError):summaries(panels["O7"],"O6")
        self.assertAlmostEqual(summaries(panels["O6"],"O6")["target_mean_belief"],f["target_mean_belief"],places=14)
        before=summaries(panels["O6"],"O6")
        m.prices[18:]*=100; m.response=1e9
        after,_=release_panels(m,cfg,310,"clean")
        self.assertEqual(before,summaries(after["O6"],"O6"))

    def test_risk_task(self):
        ce=risk_task_ce(np.array([.5,1,2,4,6.]))
        self.assertAlmostEqual(ce[1],np.sqrt(.75),places=12)
        self.assertTrue(np.all(np.diff(ce)<0))
        self.assertTrue(np.all((ce>.5)&(ce<1)))

    def test_population_supports_and_negative_dependence(self):
        cfg=Config(n=2000,belief_cash_correlation=-.6)
        p=population(cfg,np.random.default_rng(837))
        self.assertEqual(sum(p.holdings0==0),500)
        self.assertTrue(np.all((p.gamma>=.5)&(p.gamma<=6)))
        self.assertTrue(np.all((p.floor>=0)&(p.floor<=.9*p.cash0)))
        self.assertLess(np.corrcoef(p.belief_intercept,p.cash0)[0,1],0)

    def test_release_files_do_not_disclose_generating_regime(self):
        from pilot.run import run
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=run(Path(d)/"trial",replications=1,n=100,sample_size=25,stage="development")
            files=list((out/"analyst").glob("*.npz"))
            self.assertEqual(len(files),24)
            for file in files:
                self.assertNotIn("clean",file.name)
                self.assertNotIn("selective",file.name)
                with np.load(file,allow_pickle=False) as x:
                    self.assertTrue(set(x.files).isdisjoint({"seed","environment","regime","response","gamma"}))
            with (out/"analyst/features.csv").open() as f:
                fields=csv.DictReader(f).fieldnames
            self.assertTrue(set(fields).isdisjoint({"seed","environment","regime","response"}))


if __name__=="__main__":
    unittest.main(verbosity=2)
