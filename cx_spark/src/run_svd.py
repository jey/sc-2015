from pyspark import SparkContext
from pyspark import SparkConf
from randomized_svd import RandomizedSVD
from rowmatrix import RowMatrix
from sparse_row_matrix import SparseRowMatrix
from rma_utils import to_sparse
import time
import sys
import argparse
import scipy.stats
import numpy as np
import logging.config

def usage():
    print sys.exit(__doc__)

def print_params(args, logger):
    logger.info('--------------------------------------------------------')
    logger.info('---------------Computing SVD/PCA Decomposition---------------')
    logger.info('--------------------------------------------------------')
    logger.info('dataset: {0}'.format( args.dataset ) )
    logger.info('size: {0} by {1}'.format( args.dims[0], args.dims[1] ) )
    logger.info('loading file from {0}'.format( args.file_source ) )
    if args.sparse:
        logger.info('sparse format!')
    if args.center:
      logger.info('computing SVD of the row-centered data')
    if args.nrepetitions>1:
        logger.info('number of power iterations: {0}'.format( args.nrepetitions ))
        logger.info('reorthogonalizing after every {0} power iterations'.format(args.reortho))
    logger.info('number of partitions: {0}'.format( args.npartitions ))
    logger.info('rank: {0}'.format( args.k ))
    logger.info('number of iterations to run: {0}'.format( args.q ))
    logger.info('--------------------------------------------------------')
    if args.test:
        logger.info('Compute accuracies!')
    if args.save_logs:
        logger.info('Logs will be saved!')
    logger.info('--------------------------------------------------------')

class ArgumentError(Exception):
    pass

class OptionError(Exception):
    pass

def main(argv):
    logging.config.fileConfig('logging.cfg',disable_existing_loggers=False)
    logger = logging.getLogger('') #using root

    parser = argparse.ArgumentParser(description='Getting parameters.',prog='run_svd.sh')

    parser.add_argument('dataset', type=str, help='dataset.txt stores the input matrix to decompose; \
           dataset_U.txt stores left-singular vectors of the input matrix (only needed for -t); \
	   dataset_V.txt stores right-singular vectors of the input matrix (only needed for -t); \
           dataset_D.txt stores singular values of the input matrix (only needed for -t)')
    parser.add_argument('--center', help='compute SVD of the row-centered data',action='store_true')
    parser.add_argument('--dims', metavar=('m','n'), type=int, nargs=2, required=True, help='size of the input matrix')
    parser.add_argument('--sparse', dest='sparse', action='store_true', help='whether the data is sparse')
    parser.add_argument('--hdfs', dest='file_source', default='local', action='store_const', const='hdfs', help='load dataset from HDFS')
    parser.add_argument('-k', '--rank', metavar='targetRank', dest='k', default=5, type=int, help='target rank parameter')
    parser.add_argument('-q', '--niters', metavar='numIters', dest='q', default=2, type=int, help='number of power iterations to use')
    parser.add_argument('-r', '--reortho', dest='reortho', default=4, type=int, help='reorthogonalize after each set of this many power iterations')
    parser.add_argument('-c', '--cache', action='store_true', help='cache the dataset in Spark')
    parser.add_argument('-t', '--test', action='store_true', help='compute accuracies of the returned solutions')
    parser.add_argument('-s', '--save_logs', action='store_true', help='save Spark logs')
    parser.add_argument('--nrepetitions', metavar='numRepetitions', default=1, type=int, help='number of times to stack matrix vertically in order to generate large matrices')
    parser.add_argument('--npartitions', metavar='numPartitions', default=280, type=int, help='number of partitions in Spark')
    
    if len(argv)>0 and argv[0]=='print_help':
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    (m,n) = args.dims

    # validating
    if args.k > m or args.k > n:
        raise ValueError('Rank parameter({0}) should not be greater than m({1}) or n({2})'.format(args.k,m,n))

    if args.npartitions > m or args.npartitions > n:
        args.npartitions = min(m,n)

    if args.test and args.nrepetitions>1:
        raise OptionError('Do not use the test mode(-t) on replicated data(numRepetitions>1)!')

    if args.sparse and args.file_source=='hdfs':
        raise OptionError('Not yet!')

    # print parameters
    print_params(args, logger)

    # TO-DO: put these to a configuration file
    dire = '../data/'
    hdfs_dire = 'data/'
    logs_dire = 'file:///home/jiyan/cx_logs'
    logs_dire = 'file:///n/banquet/de/gittens/'

    # instantializing a Spark instance
    if args.save_logs:
        conf = SparkConf().set('spark.eventLog.enabled','true').set('spark.eventLog.dir',logs_dire)
    else:
        conf = SparkConf()
    sc = SparkContext(appName="svd_exp",conf=conf)

    # loading data
    if args.file_source=='hdfs':
        A_rdd = sc.textFile(hdfs_dire+args.dataset+'.txt',args.npartitions) #loading dataset from HDFS
    else:
        A = np.loadtxt(dire+args.dataset+'.txt') #loading dataset from local disc
        if args.sparse:
            sA = to_sparse(A)
            A_rdd = sc.parallelize(sA, args.npartitions)
        else:
            A_rdd = sc.parallelize(A.tolist(), args.npartitions)

    t = time.time()
    if args.sparse:
        matrix_A = SparseRowMatrix(A_rdd,args.dataset,m,n,args.cache) # creating a SparseRowMatrix instance
    else:
        matrix_A = RowMatrix(A_rdd,args.dataset,m,n,args.cache,repnum=args.nrepetitions) # creating a RowMatrix instance
        
    svdComputer = RandomizedSVD(matrix_A, args.center, reortho=args.reortho)
    svd = svdComputer.execute(args.k, args.q)

    if args.test:
        k = args.k
        if args.file_source != 'local':
            A = np.loadtxt(dire+args.dataset+'.txt')
        if args.center:
            U, D, V = np.linalg.svd(A - np.outer(np.ones(m), np.ones(m).dot(A)))
        else:
            U, D, V = np.linalg.svd(A,0)
        U = U[:,:k]
        D = D[:k]
        V = V[:,:k]

        _, colangles, _ = np.linalg.svd(svd[0][:, :k].T.dot(U))
        _, rowangles, _ = np.linalg.svd(svd[2][:, :k].T.dot(V))
        eigdiffs = D - svd[1][:k]

        def logarray(message, data, fmtstr='%.3f'):
            logger.info((message + ': [' + ', '.join([fmtstr]*len(data)) + ']') % tuple(data))

        logarray('actual eigenvalues', D)
        logarray('approximated eigenvalues', svd[1][:k])
	logarray('absolute differences in eigenvalues', np.abs(eigdiffs))

        logarray('angles between actual and approximate column spans', np.rad2deg(np.arccos(colangles)))
        logarray('angles between actual and approximate row spans', np.rad2deg(np.arccos(rowangles)))

if __name__ == "__main__":
    main(sys.argv[1:])



