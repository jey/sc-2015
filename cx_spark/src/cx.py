'''
CX decomposition with approximate leverage scores
'''
import logging

import numpy as np
from numpy.linalg import norm

from rma_utils import compLevExact
from randomized_svd import RandomizedSVD


logger = logging.getLogger(__name__)

class CX:
    def __init__(self, matrix_A):
        self.matrix_A = matrix_A

    def get_lev(self, k, q):
        """
        compute column leverage scores
        the code will call randomized SVD to approximate the low-rank singular subspace first
        and compute its leverage scores
        """

        rsvd = RandomizedSVD(self.matrix_A)
        U, D, V = rsvd.execute(k=k,q=q) # U*D*V.T approximates A_k

        lev_row, p_row = compLevExact(U, k=k, axis=0)
        lev_col, p_col = compLevExact(V, k=k, axis=0)

        return lev_row, lev_col, p_row, p_col

    def comp_idx(self, scheme='deterministic', r=10):
        #seleting rows based on self.lev
        #scheme can be either 'deterministic' or 'randomized'
        #r dentotes the number of rows to select
        if scheme == 'deterministic':
            self.idx = np.array(self.lev).argsort()[::-1][:r]
        elif scheme == 'randomized':
            bins = np.add.accumulate(self.p)
            self.idx = np.digitize(np.random.random_sample(r), bins)

        return self.idx

    def get_rows(self):
        #getting the selected rows back
        idx = self.idx
        rows = self.matrix_A.rdd.filter(lambda (key, row): key in idx).collect()
        self.R = np.array([row[1] for row in rows])

        return self.R.shape #shape of R is r by d

    def comp_err(self):
        #computing the reconstruction error
        Rinv = np.linalg.pinv(self.R) #its shape is d by r
        RRinv = np.dot(Rinv, self.R) #its shape is d by d
        temp = np.eye(self.matrix_A.n) - RRinv

        diff = np.sqrt( self.matrix_A.rtimes(temp,self.sc,True).map(lambda (key,row): norm(row)**2 ).sum() )

        return diff
