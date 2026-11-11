"""Small generated-input checks; no paper datasets or trained models are bundled."""
from pathlib import Path
import importlib.util
import os,sys,tempfile,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from descriptors.calculate_opt3d import opt3d_from_dm
from descriptors.atomic_weights import ATOMIC_WEIGHTS


class WorkflowTests(unittest.TestCase):
    def test_kernel_matches_literal_and_reindexing(self):
        xyz=np.random.default_rng(4).normal(size=(8,3))
        dm=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
        atoms=['m1C']*3+['m1H']+['m2C']*3+['m2H']
        actual=opt3d_from_dm(dm,atoms,nsc=12)
        expected=np.zeros((7,12))
        for i in range(7):
            for j in range(i+1,8):
                weights=np.array(ATOMIC_WEIGHTS[atoms[i][2:]])*ATOMIC_WEIGHTS[atoms[j][2:]]
                for s in range(12):
                    phase=s*.1*dm[i,j]
                    expected[:,s]+=weights*(1.0 if s==0 else np.sin(phase)/phase)
        np.testing.assert_allclose(actual,expected.ravel(),rtol=2e-13,atol=2e-13)
        order=np.array([2,0,1,3,6,4,5,7])
        np.testing.assert_allclose(actual,opt3d_from_dm(dm[np.ix_(order,order)],
                                  [atoms[i] for i in order],nsc=12),rtol=2e-13,atol=2e-13)

    def test_cluster_uses_features_not_target(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                spec=importlib.util.spec_from_file_location('cluster_test_source',ROOT/'models/opt3d_cluster_abs.py')
                base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
                x=np.array([[0,0],[.1,.1],[2,2],[2.1,2.1]],dtype=np.float32)
                d=np.column_stack([x,[0,1000,-999,1]])
                one=base.ThresholdClusterSplitter(.25,cache_dir='one');a=one.split(d)
                d[:,-1]=[9999,0,9,-8]
                two=base.ThresholdClusterSplitter(.25,cache_dir='two');b=two.split(d)
                np.testing.assert_array_equal(a[2],b[2])
                np.testing.assert_array_equal(one.group_ids_,[0,0,1,1])
                self.assertEqual(len(a[0]),2);self.assertEqual(len(a[1]),2)
                self.assertEqual(set(a[2]),{0,2})
            finally:os.chdir(previous)


    def test_dm_permutations_preserve_distances_and_label(self):
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                spec=importlib.util.spec_from_file_location('dm_test_source',ROOT/'models/dm_random_signed.py')
                base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
                base.set_data_random_seed(0)
                atoms=['m1C']*3+['m1H']+['m2C']*3+['m2H']
                xyz=np.random.default_rng(8).normal(size=(8,3))
                dm=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).astype(np.float32)
                rows=np.concatenate([dm.ravel(),[-23.5]]).astype(np.float32)[None,:]
                augmented=base.DataProcessor(total_perms=4).generate_permutation_data(rows,atoms)
                self.assertEqual(augmented.shape,(5,65))
                np.testing.assert_array_equal(augmented[:,-1],np.full(5,-23.5))
                for row in augmented:
                    matrix=row[:-1].reshape(8,8)
                    np.testing.assert_allclose(matrix,matrix.T)
                    np.testing.assert_allclose(np.sort(matrix.ravel()),np.sort(dm.ravel()))
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
