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
        logger.info('Finish doing gaussian_projection!')

        for i in range(q):
            logger.info('Computing leverage scores, at iteration {0}!'.format(i+1))
            print 'Computing leverage scores, at iteration {0}!'.format(i+1)
            if i % reo == reo-1:
                logger.info("Reorthogonalzing!")
                Q, R = np.linalg.qr(Y)
                Y = Q
                logger.info("Done reorthogonalzing!")
            Y = self.matrix_A.atamat(Y)
            logger.info('Finish iteration {0}!'.format(i+1))
            print 'Finish iteration {0}!'.format(i+1)

        # shape of Y is n by 2k

        Q, R = np.linalg.qr(Y) # shape of Q is n by 2k

        B = self.matrix_A.rtimes(Q).T
        #B = np.dot(Q.T,A.T) # shape of B is 2k by m

        U_hat,D,V = np.linalg.svd(B,full_matrices=0)
        U = np.dot(Q,U_hat)

        return U, D, V

