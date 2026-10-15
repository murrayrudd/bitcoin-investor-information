"""Tests of inferential guarantees, information restrictions, and acceleration."""
import unittest
from dataclasses import replace
import numpy as np
from pilot.config import Config
from pilot.economy import simulate
from pilot.kernel import enable_compiled,disable_compiled
from pilot.observations import release_panels
from pilot.inference_features import summaries
from pilot.inference_design import candidate_catalog,truth_cases,seed_bank,N_TRAIN,N_CAL,N_REF,N_EVAL,N_ORACLE
from pilot.inference import rank_pvalues,pooled_precision


class InferenceTests(unittest.TestCase):
    def test_exact_rank_rejection_and_ties(self):
        # For every possible ordering of one test draw and 99 exchangeable
        # calibration draws, a .04 test rejects exactly four ranks.
        p=[]
        for observed in range(100):
            calibration=np.delete(np.arange(100),observed)[None,:]
            p.append(rank_pvalues(calibration,np.array([observed]))[0])
        self.assertEqual(sum(x<=.04 for x in p),4)
        self.assertEqual(min(p),.01)
        np.testing.assert_array_equal(rank_pvalues(np.ones((2,99)),np.ones(2)),[1,1])

    def test_seed_banks_disjoint(self):
        seen=set()
        for row in candidate_catalog():
            for role,n in [('train',N_TRAIN),('calibration',N_CAL),('reference',N_REF)]:
                seeds=set(seed_bank(role,row['candidate_id'],n))
                self.assertFalse(seen&seeds);seen.update(seeds)
        for row in truth_cases():
            for role,n in [('evaluation',N_EVAL),('oracle',N_ORACLE)]:
                seeds=set(seed_bank(role,row['truth_id'],n))
                self.assertFalse(seen&seeds);seen.update(seeds)
        self.assertFalse(seen & {s+1_000_000_000 for s in seen})
        self.assertTrue(min(seen)>450007)

    def test_kernel_and_release_links(self):
        for i,changes in enumerate([{},dict(belief_drift=-.03,risk_median=3.4),dict(allowance_growth=-.02,commitment_median=.75),dict(funding_rate=.01,belief_cash_correlation=-.2)]):
            cfg=replace(Config(),n=400,sample_size=100,**changes)
            disable_compiled();reference=simulate(cfg,910001+i)
            enable_compiled(required=True);fast=simulate(cfg,910001+i)
            np.testing.assert_allclose(reference.prices,fast.prices,atol=1e-10,rtol=0)
            self.assertAlmostEqual(reference.response,fast.response,places=8)
            panels,_=release_panels(fast,cfg,950000+i,'selective')
            a=summaries(panels['O6'],'O6');b=summaries(panels['O7'],'O7')
            for key in b:self.assertAlmostEqual(a[key],b[key],places=12)
            # Independent module permutations must leave all O7 summaries fixed.
            data={k:v.copy() for k,v in panels['O7'].items()}
            data['belief']=data['belief'][:,::-1]
            data['target_belief']=data['target_belief'][:,::-1]
            np.testing.assert_allclose(list(b.values()),list(summaries(data,'O7').values()),atol=1e-12,rtol=0)
            for p in panels:
                self.assertFalse({'gamma','response','environment','regime','seed','floor0'}&set(panels[p]))

    def test_covariance_uses_within_candidate_noise(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(4,32,5))
        a=pooled_precision(x,np.arange(4))
        b=pooled_precision(x+np.arange(4)[:,None,None]*100,np.arange(4))
        np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertTrue(np.linalg.eigvalsh(a).min()>0)


if __name__=='__main__':unittest.main()
