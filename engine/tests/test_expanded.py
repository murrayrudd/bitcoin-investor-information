"""Tests target seed isolation, analytic derivatives, and search correctness."""
import unittest
import json,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from pilot.expanded_design import *
from pilot.expanded_fit import basis,POLY,rank_threshold,ContinuousFitter,POWERS

class ExpandedTests(unittest.TestCase):
    def test_parameter_roundtrip_and_response_participation_separation(self):
        rng=np.random.default_rng(19)
        for x in rng.uniform(-1,1,(10,9)):
            np.testing.assert_allclose(encode(decode(x)),x,atol=1e-12)
        x,s=design_points(1000,9,iid=True)
        zero=encode({})
        for i in range(1,4):
            for d in (5,6,7):
                if d!=4+i:np.testing.assert_array_equal(x[s==i,d],zero[d])
    def test_polynomial_gradient_at_interior_and_zero(self):
        for x in [np.zeros(9),np.random.default_rng(2).uniform(-1,1,9)]:
            value,grad=basis(x,True)
            np.testing.assert_allclose(value,POLY.transform(x[None,:])[0],atol=1e-14)
            numerical=approx_derivative(lambda z:basis(z),x)
            np.testing.assert_allclose(grad.T,numerical,atol=1e-9)
    def test_rank_quantiles_include_finite_sample_correction(self):
        self.assertEqual(rank_threshold(np.arange(1999),.035),1929)
        self.assertEqual(rank_threshold(np.arange(199),.01),197)
        self.assertTrue(np.isinf(rank_threshold(np.arange(9),.01)))
    def test_complete_seed_sets_disjoint(self):
        seen=set()
        for b in range(2):
            for role,n,reps in [('train',1024,8),('check',128,12),('calibration',1999,1),('response',199,40)]:
                values={seed(b,role,i,j) for i in range(n) for j in range(reps)}
                self.assertEqual(len(values),n*reps);self.assertFalse(seen&values);seen.update(values)
        for role,reps in [('evaluation',40),('oracle',640)]:
            values={seed(0,role,i,j) for i in range(12) for j in range(reps)}
            self.assertFalse(seen&values);seen.update(values)
        self.assertFalse(seen&{v+2_000_000_000 for v in seen})
        self.assertGreater(min(seen),100_000_000)
    def test_continuous_search_matches_known_ellipsoid_endpoints(self):
        coef=np.zeros((len(POWERS),9));response=np.zeros(len(POWERS))
        for d in range(9):
            row=np.flatnonzero((POWERS[:,d]==1)&(POWERS.sum(axis=1)==1))[0]
            coef[row,d]=1
            if d==0:response[row]=1
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'
            np.savez(path,coef=coef,white=np.eye(9),response_coef=response,
              train_x=np.zeros((1,9)),metadata=np.array(json.dumps(dict(cutoff=.25,response_error_pp=.03,panel='O6'))))
            model=ContinuousFitter(path)
            fits=model.fit(np.zeros(9),budgets=(2,8))
            for f in fits:
                self.assertFalse(f['empty'])
                self.assertAlmostEqual(f['lower'],-.53,places=6)
                self.assertAlmostEqual(f['upper'],.53,places=6)
            self.assertGreaterEqual(fits[1]['width'],fits[0]['width']-1e-12)
            # A distant release has no feasible parameter in the bounded box.
            self.assertTrue(model.fit(np.full(9,3.),budgets=(2,))[0]['empty'])

if __name__=='__main__':unittest.main()
