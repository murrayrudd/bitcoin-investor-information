"""The refinement must find an analytically known disconnected branch."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pilot.repair_fit import POWERS, RepairedFitter
from pilot.repair_refine import refine

class RefinementTests(unittest.TestCase):
    def test_disconnected_feasible_regions_and_monotone_expansion(self):
        coef=np.zeros((len(POWERS),9)); response=np.zeros(len(POWERS)); cutoff=np.zeros(len(POWERS))
        cutoff[0]=np.log(.0025)
        for d in range(9):
            p=np.zeros(9,dtype=int);p[d]=2 if d==0 else 1
            i=np.flatnonzero(np.all(POWERS==p,axis=1))[0];coef[i,d]=1.
        p=np.zeros(9,dtype=int);p[0]=1
        response[np.flatnonzero(np.all(POWERS==p,axis=1))[0]]=1.
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'
            np.savez(path,coef=coef,white=np.eye(9),response_coef=response,cutoff_coef=cutoff,
                     train_x=np.zeros((1,9)),metadata=np.array(json.dumps(dict(panel='O1',response_error_pp=.03))))
            model=RepairedFitter(path);y=np.zeros(9);y[0]=.49
            xlo=np.zeros(9);xhi=np.zeros(9);xlo[0]=np.sqrt(.44);xhi[0]=np.sqrt(.54)
            initial=dict(empty=False,lower=xlo[0]-.03,upper=xhi[0]+.03,
                         width=xhi[0]-xlo[0]+.06,lower_x=xlo.tolist(),upper_x=xhi.tolist())
            fixed=refine(model,y,initial)
            self.assertAlmostEqual(fixed['lower'],-np.sqrt(.54)-.03,places=6)
            self.assertAlmostEqual(fixed['upper'],np.sqrt(.54)+.03,places=6)
            self.assertLessEqual(fixed['lower'],initial['lower'])
            self.assertGreaterEqual(fixed['upper'],initial['upper']-1e-10)
            self.assertLessEqual(model.quantities(fixed['lower_x'],y)['ratio'],1+1e-7)

if __name__=='__main__':unittest.main()
