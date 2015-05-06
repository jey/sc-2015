import sys
sys.path.append('../src/')
import numpy as np
import unittest
from randomized_svd import RandomizedSVD
from sparse_row_matrix import SparseRowMatrix
from rma_utils import to_sparse

class RandomizedSVDTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)

    def test_rpca(self):
        rpca = RandomizedSVD(self.matrix_A)
        U,D,V = rpca.execute(k=10,q=5)

        print U.shape, V.shape, D.shape

        print np.dot( U.T, U )

        self.assertTrue( U.shape == (1000,20) )

class MatrixMultiplicationTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_A = SparseRowMatrix(sparse_matrix_rdd,'test_data',1000,100)
        self.matrix_A2 = SparseRowMatrix(sparse_matrix_rdd2,'test_data',100,1000)

    def test_mat_rtimes(self):
        mat = np.random.rand(100,50)
        p = self.matrix_A.rtimes(mat)
        p_true = np.dot( A, mat )
        self.assertTrue( np.linalg.norm(p-p_true)/np.linalg.norm(p_true) < 1e-5 )

loader = unittest.TestLoader()
suite_list = []
suite_list.append( loader.loadTestsFromTestCase(RandomizedSVDTestCase) )
suite_list.append( loader.loadTestsFromTestCase(MatrixMultiplicationTestCase) )
suite = unittest.TestSuite(suite_list)

if __name__ == '__main__':
    from pyspark import SparkContext

    A = np.loadtxt('../data/unif_bad_1000_100.txt')
    A2 = np.loadtxt('../data/unif_bad_100_1000.txt')
    sA = to_sparse(A)
    sA2 = to_sparse(A2)

    sc = SparkContext(appName="rpca_test_exp")

    matrix_rdd = sc.parallelize(A.tolist(),140)
    matrix_rdd2 = sc.parallelize(A2.tolist(),20)
    sparse_matrix_rdd = sc.parallelize(sA,140)  # sparse_matrix_rdd has records in (row,col,val) format
    sparse_matrix_rdd2 = sc.parallelize(sA2,50) 

    runner = unittest.TextTestRunner(stream=sys.stderr, descriptions=True, verbosity=1)
    runner.run(suite)