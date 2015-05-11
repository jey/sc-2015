'''
Randomized SVD
'''
import logging

import numpy as np
from scipy.sparse import coo_matrix
from numpy.linalg import norm
from sparse_row_matrix import SparseRowMatrix
from operator import add

logger = logging.getLogger(__name__)

class RandomizedSVD:
    def __init__(self, matrix_A, centered=False, reortho=4):
        self.matrix_A = matrix_A
	self.centered = centered
        self.reo = reortho

    def execute(self, k, q):

        logger.info('Ready to do gaussian_projection!')

	# use slightly more probes if are going to use a rank-one update to
	# approximate the PCA
	if self.centered:
            ell = 2*k+2
        else:
            ell = 2*k
        if isinstance(self.matrix_A, SparseRowMatrix):
            Y = self.matrix_A.gaussian_projection(ell).T
        else: # RowMatrix doesn't have a gaussian_projection routine, so materialize an explicit matrix FIX!
            Omega = np.random.rand(self.matrix_A.m, ell) # since we are computing A.T*Omega
            Y = self.matrix_A.ltimes(Omega) # neither rtimes or ltimes work, and I think this should be ltimes
        logger.info('Finished doing gaussian_projection!')

        for i in range(q):
            logger.info('Comuting randomized SVD, at iteration {0}!'.format(i+1))
            print 'Computing randomized SVD, at iteration {0}!'.format(i+1)
            if i % self.reo == self.reo-1:
                logger.info("Reorthogonalzing!")
                Q, R = np.linalg.qr(Y)
                Y = Q
                logger.info("Done reorthogonalzing!")
            Y = self.matrix_A.atamat(Y)
            logger.info('Finished iteration {0}!'.format(i+1))
            print 'Finished iteration {0}!'.format(i+1)

        # shape of Y is n by 2k

        Q, R = np.linalg.qr(Y) # shape of Q is n by 2k

        logger.info('Ready to do rtimes!')
        B = self.matrix_A.rtimes(Q).T
        logger.info('Finished rtimes!')
        #B = np.dot(Q.T,A.T) # shape of B is 2k by m

        U_hat,D,V = np.linalg.svd(B,full_matrices=0)
        U = np.dot(Q,U_hat)

        # this function is actually finding SVD for A.T
        # so we need to swap U and V
        temp = U
        U = V.T
        V = temp # U * D * V.T will be a good approximation to A_k

	if self.centered:
	  # turn SVD(X) into SVD(X - 11^TX)
	  logger.info('Using a rank-one update to modify the SVD of the \
	  uncentered data to that of the uncentered data')
	  
	  rowmean = np.asarray([0]*self.matrix_A.n)

	  # how we compute the mean depends on the type of matrix:
	  # SparseRowMatrix or RowMatrix

          n = self.matrix_A.n
	  if isinstance(self.matrix_A, SparseRowMatrix):

	    def addvecs(numpyvec, sparsevecspec):
	      tempmat = coo_matrix(( sparsevecspec[1][1], ( [0]*len(sparsevecspec[1][0]), sparsevecspec[1][0])), shape=(1, n)).tocsr()
	      numpyvec = numpyvec + np.asarray(tempmat).flatten()
	      return numpyvec

	  elif isinstance(self.matrix_A, RowMatrix):

	    def addvecs(numpyvec, rowvec):
	      numpyvec = numpyvec + np.asarray(rowvec[1])
	      return numpyvec
          
	  self.matrix_A.rdd.aggregate(rowmean, addvecs, add)
	  rowmean /= self.matrix_A.m

          logger.info('Computed row mean of data matrix')

	  x = -1*np.sum(U.T, axis=1)
	  y = V.T.dot(rowmean)
	  residx = -1*np.ones((self.matrix_A.m,)) - U.dot(x)
	  residy = rowmean - V.dot(y)
	  norm_residx = norm(residx) 
	  norm_residy = norm(residy)

          logger.info('Norms of residuals of 1 and rowmean w.r.t the col and row spaces: {0}, {1}'.format(norm_residx, norm_residy))

# TODO: of course the residual of the rowmean w.r.t. the row space is 0, so just automatically use a random vector

	  # if residuals are essentially zero, then use the residuals of
	  # normalized random vectors
	  EPS = 10e-11
          if norm_residx < EPS:
            randv = np.random.rand(self.matrix_A.m)
	    x = U.T.dot(randv)
	    residx = randv - U.dot(x)
	    norm_residx = 0
	  if norm_residy < EPS:
            randv = np.random.rand(self.matrix_A.n)
	    y = V.T.dot(randv)
	    residy = randv - V.dot(y)
	    norm_residy = 0

          # If ell were large enough to justify, could use scipy svds + LinearOperator to avoid forming matrices explicitly
	  Uhat, D, Vhat = np.linalg.svd(np.diag(np.append(D, 0)) + 
	      np.outer(np.append(x, norm_residx), np.append(y, norm_residy)))

	  U = np.c_[U, residx/norm(residx)].dot(Uhat)
	  V = np.c_[V, residy/norm(residy)].dot(Vhat)

	  # note that this is a rank ell+1 decomposition, and that we potentially
	  # have a spurious 0 eigenvalue (if either of the residuals were zero)
          # so truncate down by 1 eigenvalue (not to k, since the noncentered version can also return more than k eigenpairs)
          truncatedrank = len(D)-1
          U = U[:,:truncatedrank]
          D = D[:truncatedrank]
          V = V[:,:truncatedrank]

        logger.info('Finished randomized SVD!')

        return U, D, V
