"""Independent mathematical and isolation checks for version 0.4."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from pilot.repair_fit import POLY, POWERS, basis, RepairedFitter, regression, fit_log_quantile
from pilot.repair_design import seed, cases, COUNTS, draw
from pilot.repair_bank import jobs

class RepairTests(unittest.TestCase):
    def model(self, folder, variable_cutoff=False):
        coef=np.zeros((len(POWERS), 9)); response=np.zeros(len(POWERS)); cutoff=np.zeros(len(POWERS))
        for d in range(9):
            i=np.flatnonzero((POWERS[:, d] == 1) & (POWERS.sum(axis=1) == 1))[0]
            coef[i, d]=1.
            if d == 0:
                response[i]=1.
                if variable_cutoff:
                    cutoff[i]=.3
        cutoff[0]=np.log(.25)
        path=Path(folder)/'test.npz'
        np.savez(path, coef=coef, white=np.eye(9), response_coef=response,
                 cutoff_coef=cutoff, train_x=np.zeros((1, 9)),
                 metadata=np.array(json.dumps(dict(panel='O6', response_error_pp=.03))))
        return RepairedFitter(path)

    def test_quartic_basis_and_analytic_derivatives(self):
        for x in [np.zeros(9), np.ones(9), np.random.default_rng(56).uniform(-1, 1, 9)]:
            v, g=basis(x, True)
            np.testing.assert_allclose(v, POLY.transform(x[None, :])[0], atol=1e-14)
            np.testing.assert_allclose(g.T, approx_derivative(basis, x), atol=2e-9)

    def test_variable_cutoff_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            model=self.model(folder, True); y=np.zeros(9); x=np.linspace(-.3, .3, 9)
            analytical=model.quantities(x, y, True)['ratio_grad']
            numerical=approx_derivative(lambda xx: model.quantities(xx, y)['ratio'], x).ravel()
            np.testing.assert_allclose(analytical, numerical, atol=2e-8)

    def test_ellipsoid_endpoints_and_nested_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            model=self.model(folder); fits=model.fit(np.zeros(9), budgets=(2, 6))
            for row in fits:
                self.assertFalse(row['empty'])
                self.assertAlmostEqual(row['lower'], -.53, places=6)
                self.assertAlmostEqual(row['upper'], .53, places=6)
                self.assertLessEqual(model.quantities(row['lower_x'], np.zeros(9))['ratio'], 1+1e-7)
            self.assertGreaterEqual(fits[1]['width'], fits[0]['width']-1e-10)
            self.assertTrue(model.fit(np.full(9, 3.), budgets=(2,))[0]['empty'])

    def test_response_regression_cannot_use_selection(self):
        rng=np.random.default_rng(34); x=rng.uniform(-1, 1, (200, 9))
        y=2+x[:, 0]+x[:, 0]**2+.1*x[:, 8]
        coef=regression(x, y, np.ones(200), 3, participation=False)
        self.assertTrue(np.all(coef[POWERS[:, 8] != 0] == 0))
        a=np.zeros(9); b=a.copy(); b[-1]=1
        self.assertAlmostEqual(basis(a)@coef, basis(b)@coef, places=14)

    def test_quantile_regression_recovers_heteroscedastic_score_scale(self):
        rng=np.random.default_rng(55); x=rng.uniform(-1, 1, (180, 9))
        targets=3+.6*x[:, :1]+rng.normal(0, .15, (180, 80))
        coef, diagnostic=fit_log_quantile(x, targets)
        a=np.zeros(9); b=a.copy(); a[0]=-1; b[0]=1
        self.assertTrue(diagnostic['success'])
        self.assertAlmostEqual(basis(b)@coef-basis(a)@coef, 1.2, delta=.15)

    def test_seed_namespaces_and_unseen_settings(self):
        ranges=[]
        for bank in (0, 1):
            for role, n, reps in [('train', 812, 32), ('check', 96, 16),
                                   ('quantile', 384, 24), ('calibration', 1797, 1), ('response', 297, 32)]:
                ranges.append({seed(bank, role, i, r) for i in range(n) for r in range(reps)})
        for role, n, reps in [('evaluation', 24, 80), ('oracle', 24, 512), ('audit', 192, 128)]:
            ranges.append({seed(0, role, i, r) for i in range(n) for r in range(reps)})
        seen=set()
        for values in ranges:
            self.assertFalse(seen & values); seen.update(values)
        self.assertTrue(min(seen) >= 600_000_000)
        self.assertFalse(seen & {s+2_000_000_000 for s in seen})
        c=cases(); self.assertEqual(len(c), 24)
        old={json.dumps(r['parameters'], sort_keys=True) for r in c[:12]}
        self.assertFalse(old & {json.dumps(r['parameters'], sort_keys=True) for r in c[12:]})

    def test_validation_requires_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                list(jobs(folder, '.', 'validation'))

if __name__ == '__main__':
    unittest.main()
