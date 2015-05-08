'''
Randomized SVD
'''
import logging

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)

class RandomizedSVD:
    def __init__(self, matrix_A):
        self.matrix_A = matrix_A

    def execute(self, k, q):
        reo = 4

        logger.info('Ready to do gaussian_projection!')
        Y = self.matrix_A.gaussian_projection(2*k).T
        logger.info('Finished doing gaussian_projection!')

        for i in range(q):
            logger.info('Comuting randomized SVD, at iteration {0}!'.format(i+1))
            print 'Computing randomized SVD, at iteration {0}!'.format(i+1)
            if i % reo == reo-1:
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
        logger.info('Finished ritmes!')
        #B = np.dot(Q.T,A.T) # shape of B is 2k by m

        U_hat,D,V = np.linalg.svd(B,full_matrices=0)
        U = np.dot(Q,U_hat)

        # this function is actually finding SVD for A.T
        # so we need to swap U and V
        temp = U
        U = V.T
        V = temp # U * D * V.T will be a good approximation to A_k

        logger.info('Finished randomized SVD!')

        return U, D, V

