import unittest
import numpy as np

class Checks(unittest.TestCase):
    def test_fixed_target_line_needs_only_one_nonzero_probe(self):
        parts=[np.array([[0.,0.]]),np.array([[2.,0.]])]
        self.assertEqual(np.linalg.matrix_rank(np.vstack(parts)),1)
        self.assertTrue(any(np.linalg.matrix_rank(p)==1 for p in parts))
    def test_order_target_containment_matters(self):
        first=np.array([[1.,0.]])
        second=np.array([[1.,0.],[0.,2.]])
        self.assertEqual(np.linalg.matrix_rank(first),1)
        self.assertEqual(np.linalg.matrix_rank(second),2)
    def test_full_rank_does_not_bound_finite_remainder(self):
        k,u,M=3,0.75,20.
        f=lambda x:0.5*x*x+M*x**(k+1)
        self.assertAlmostEqual(f(u)-f(0.)-u*u/2.,M*u**(k+1))
    def test_alias_pair_minimax(self):
        S=[-1.,1.]; T=lambda s:s*s; H=lambda s:s
        gamma=max(abs(H(a)-H(b)) for a in S for b in S if T(a)==T(b))
        self.assertEqual(gamma,2.)
        self.assertEqual(max(abs(H(s)) for s in S),gamma/2.)
    def test_conditioned_local_bound(self):
        T=lambda x:2*x+0.1*x*x
        eps=0.09; xs=np.linspace(-1,1,201)
        gamma=max(abs(a-b) for a in xs for b in xs if abs(T(a)-T(b))<=2*eps+1e-12)
        self.assertLessEqual(gamma,2*eps/1.8+1e-10)

if __name__=="__main__":
    unittest.main()
