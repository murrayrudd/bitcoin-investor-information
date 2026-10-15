import json,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.stats import binom,beta
from pilot.continuation_fit import DirectedFitter
from pilot.continuation_calibration import tolerance_rank
from pilot.continuation_design import seed,specification
from pilot.repair_fit import POWERS

class ContinuationTests(unittest.TestCase):
    def test_heterogeneous_bound_against_exact_poisson_binomial_probabilities(self):
        from scripts.continuation_summary import heterogeneous_lower
        for probs in [np.linspace(.1,.95,12),np.array([.99]*6+[.35]*6),np.full(12,.8)]:
            mass=np.array([1.])
            for p in probs:mass=np.convolve(mass,[1-p,p])
            for alpha in (.05,.2):
                reject=sum(mass[k] for k in range(13) if heterogeneous_lower(k,12,alpha)>np.mean(probs))
                self.assertLessEqual(reject,alpha+1e-12)
                self.assertAlmostEqual(heterogeneous_lower(12,12,alpha),alpha**(1/12),places=12)

    def test_tolerance_order_statistic_has_prespecified_confidence(self):
        n=512;k=tolerance_rank(n)
        self.assertGreaterEqual(binom.cdf(k-1,n,.965),1-.05/96)
        self.assertLess(binom.cdf(k-2,n,.965),1-.05/96)
        self.assertAlmostEqual(1-beta.cdf(.965,k,n+1-k),binom.cdf(k-1,n,.965),places=12)
        with self.assertRaises(ValueError):tolerance_rank(10)

    def test_disconnected_feasible_set_recovers_both_known_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            a=np.zeros((len(POWERS),1));r=np.zeros(len(POWERS));c=np.zeros(len(POWERS));a[0,0]=-.64
            linear=np.flatnonzero((POWERS[:,0]==1)&(POWERS.sum(axis=1)==1))[0]
            square=np.flatnonzero((POWERS[:,0]==2)&(POWERS.sum(axis=1)==2))[0]
            a[square,0]=1;r[linear]=1;c[0]=np.log(.01)
            path=Path(folder)/'model.npz';np.savez(path,coef=a,response_coef=r,cutoff_coef=c,white=np.eye(1),
                train_x=np.zeros((1,9)),metadata=np.array(json.dumps(dict(panel='O1',response_error_pp=.03))))
            m=DirectedFitter(path);fits=m.fit(np.zeros(1),budgets=(4,8))
            for f in fits:
                self.assertAlmostEqual(f['lower'],-np.sqrt(.74)-.03,places=6)
                self.assertAlmostEqual(f['upper'],np.sqrt(.74)+.03,places=6)
                self.assertLessEqual(m.quantities(f['lower_x'],np.zeros(1))['ratio'],1+1e-7)
            g=m.global_check(np.zeros(1),fits[-1],iterations=4,population=16)
            self.assertLessEqual(g['lower'],fits[-1]['lower']+1e-12)
            self.assertGreaterEqual(g['upper'],fits[-1]['upper']-1e-12)

    def test_new_seed_namespaces_are_disjoint(self):
        sets=[]
        for b in (0,1):
            for role in ('calibration','validation','oracle','audit'):
                sets.append({seed(b,role,i,r) for i in range(72) for r in range(512)})
        used=set()
        for s in sets:self.assertFalse(s&used);used.update(s)
        self.assertGreater(min(used),3_000_000_000)
        self.assertFalse(used&{s+2_000_000_000 for s in used})

    def test_design_keeps_known_failures_and_new_locations(self):
        s=specification();self.assertEqual(len(s['anchors']),24);self.assertEqual(len(s['cases']),36)
        focal={r['original_audit_index'] for r in s['cases'] if r['kind']=='focal_endpoint'}
        self.assertTrue(focal<={r['old_audit_index'] for r in s['anchors']})
        self.assertEqual(sum(r['status']=='new_location' for r in s['cases']),9)

if __name__=='__main__':unittest.main()
