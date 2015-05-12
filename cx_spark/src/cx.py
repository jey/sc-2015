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

        return lev_row, lev_col, p_row, p_col

    def comp_idx(self, p, scheme='deterministic', r=10):
        #seleting rows based on self.lev
        #scheme can be either 'deterministic' or 'randomized'
        #r dentotes the number of rows to select

        if scheme == 'deterministic':
            self.idx = np.array(p).argsort()[::-1][:r]
        elif scheme == 'randomized':
            bins = np.add.accumulate(p)
            self.idx = np.digitize(np.random.random_sample(r), bins)

        return self.idx

    def get_rows(self):
        #getting the selected rows back
        idx = self.idx
        rows = self.matrix_A.rdd.filter(lambda (key, row): key in idx).collect()
        self.R = np.array([row[1] for row in rows])

        return self.R.shape #shape of R is r by d

    def comp_err(self, R=None):
        #computing the reconstruction error
        #Rinv = np.linalg.pinv(self.R) #its shape is d by r
        #RRinv = np.dot(Rinv, self.R) #its shape is d by d
        #temp = np.eye(self.matrix_A.n) - RRinv

        n = self.matrix_A.n

        if R is None:
            idx = self.idx
            rows = self.matrix_A.rdd.filter(lambda (key, row): key in idx).collect()
            #self.R = np.array([row[1] for row in rows])

            data = {'row':[], 'col':[], 'val':[]}
            row_idx = 0
            for row in rows:
                data['row'] += [row_idx]*len(row[1][0])
                data['col'] += row[1][0].tolist()
                data['val'] += row[1][1].tolist()
                row_idx += 1

            R = form_csr_matrix(data,len(rows),n).toarray()
            np.savetxt('temp_R.txt', R)

        print R.shape

        # assuming the matrix is tall
        U, D, V = np.linalg.svd(R, full_matrices=False)

        print V.shape # shape of V should be r by d

        V = self.matrix_A.rdd.context.broadcast(V)

        #diff = np.sqrt( self.matrix_A.rtimes(temp,self.sc,True).map(lambda (key,row): norm(row)**2 ).sum() )
        #sum_norms = self.matrix_A.rdd.map(lambda row: comp_norm_mapper(row, V.value, n)).sum()
        cnm = comp_norm_mapper()
        sum_norms = self.matrix_A.rdd.mapPartitions(lambda records: cnm(records, V=V.value, n=n)).sum()

        # do it in mappartitions?

        return sum_norms

#def comp_norm_mapper(row, V, n):
#    arr = np.zeros(n)
#    arr[row[1][0]] = row[1][1] # densify the row
#
#    a = np.linalg.norm(arr)**2
#    b = np.linalg.norm(arr - np.dot( np.dot(arr, V.T), V ))**2
#
#    return np.array([a,b])
    #return a

class comp_norm_mapper(BlockMapper):

    def __init__(self):
        BlockMapper.__init__(self, 500)
        self.data = {'row':[],'col':[],'val':[]}
        self.results = np.zeros(2)

    def parse(self, r):
        self.keys.append(r[0])
        self.data['row'] += [self.sz]*len(r[1][0])
        self.data['col'] += r[1][0].tolist()
        self.data['val'] += r[1][1].tolist()

    def process(self, V, n):

        data = form_csr_matrix(self.data,len(self.keys),n).toarray()

        a = np.linalg.norm(data, 'fro')**2
        b = np.linalg.norm( np.dot( np.dot(data, V.T), V), 'fro')**2
        b = 0

        self.results += np.array([a,b])

        return iter([])

    def close(self):
        yield self.results
        #for r in emit_results(self.atamat, self.direct_sum, axis=0):
        #    yield r

