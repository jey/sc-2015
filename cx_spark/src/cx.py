'''
CX decomposition with approximate leverage scores
'''
import logging

import numpy as np
from numpy.linalg import norm

from rma_utils import compLevExact
from randomized_svd import RandomizedSVD
from rma_utils import form_csr_matrix
from utils import BlockMapper

logger = logging.getLogger(__name__)

class CX:
    def __init__(self, matrix_A):
        self.matrix_A = matrix_A

    def get_lev(self, k, q):
        """
        compute column leverage scores
        the code calls randomized SVD to approximate the low-rank singular subspace first
        and computes leverage scores of the approximation
        """

        rsvd = RandomizedSVD(self.matrix_A)
        U, D, V = rsvd.execute(k=k,q=q) # U*D*V.T approximates A_k

        # computing leverage scores of the approximation of A_k
        logger.info("Ready to compute leverage scores of the approximation!")
        lev_row, p_row = compLevExact(U, k=k, axis=0)
        lev_col, p_col = compLevExact(V, k=k, axis=0)
        logger.info("Finished computing leverage scores!")

        return lev_row, lev_col, p_row, p_col, U, D, V

    def comp_idx(self, p, scheme='deterministic', r=10):
        #seleting rows based on p
        #scheme can be either 'deterministic' or 'randomized'
        #r dentotes the number of rows to select

        if scheme == 'deterministic':
            self.idx = np.array(p).argsort()[::-1][:r]
        elif scheme == 'randomized':
            bins = np.add.accumulate(p)
            self.idx = np.digitize(np.random.random_sample(r), bins)

        return self.idx

    def get_rows(self, idx):

        n = self.matrix_A.n

        #getting the selected rows back
        rows = self.matrix_A.rdd.filter(lambda (key, row): key in idx).collect()
        
        #self.R = np.array([row[1] for row in rows])
        data = {'row':[], 'col':[], 'val':[]}
        i = 0
        for row in rows:
            data['row'] += [i]*len(row[1][0])
            data['col'] += row[1][0].tolist()
            data['val'] += row[1][1].tolist()
            i += 1

        R = form_csr_matrix(data,len(rows),n).toarray()

        print R.shape

        return R  #shape of R is r by d

    def comp_err(self, R, Va):
        #computing the reconstruction error

        n = self.matrix_A.n
        print R.shape

        # assuming the matrix is tall
        Ur, Dr, Vr = np.linalg.svd(R, full_matrices=False)
        Vr = Vr.T

        print Vr.shape # shape of V should be d by r
        Vr = self.matrix_A.rdd.context.broadcast(Vr)
        Va = self.matrix_A.rdd.context.broadcast(Va)

        cnm = comp_norm_mapper()
        sum_norms = self.matrix_A.rdd.mapPartitions(lambda records: cnm(records, Vr=Vr.value, Va=Va.value, n=n)).sum()

        relative_err_c = np.sqrt( sum_norms[1]/sum_norms[0] )
        relative_err_k = np.sqrt( sum_norms[2]/sum_norms[0] )

        return relative_err_c, relative_err_k

class comp_norm_mapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self, 500)
        self.data = {'row':[],'col':[],'val':[]}
        self.results = np.zeros(3)

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, Vr, Va, n):

        data = form_csr_matrix(self.data,len(self.keys),n).toarray()

        a = np.linalg.norm(data, 'fro')**2
        b = np.linalg.norm( data - np.dot( np.dot(data, Vr), Vr.T), 'fro')**2
        c = np.linalg.norm( data - np.dot( np.dot( np.dot( np.dot(data,Va), Va.T), Vr), Vr.T), 'fro')**2

        self.results += np.array([a,b,c])

        return iter([])

    def close(self):
        yield self.results

