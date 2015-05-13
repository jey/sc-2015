from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel 
from utils import prepare_matrix
from cx import CX
from sparse_row_matrix import SparseRowMatrix

import os
import ast
import numpy as np
import logging.config
import logging

logging.config.fileConfig('logging.cfg', disable_existing_loggers=False) 
logger = logging.getLogger(__name__)

conf = SparkConf().set('spark.eventLog.enabled','true')
#.set('spark.driver.maxResultSize', '15g').set("spark.executor.memory", "30g")
sc = SparkContext(appName='samp_eval_err',conf=conf)

def parse(string):
    s = str(string)
    val = ast.literal_eval(s)
    return val[0], (np.array(val[1][0]), np.array(val[1][1]))




logger.info('----------------------------------------')
logger.info('Ready to do CX (sampling and computing reconstruction error)!')

data = sc.textFile('msi_data',500).map(lambda x:parse(x))

row_shape = 8258911
column_shape = 131048

matrix_A = SparseRowMatrix(data, 'output', row_shape, column_shape, False)

#row_lev = np.loadtxt( '../results/row_leverage_scores_logged' )

logger.info('Loading p values!')
row_p = np.loadtxt( '../results/row_p_scores_logged' )
logger.info('Finish loading! Its shape is {0}'.format(row_p.shape))

cx = CX(matrix_A)

scheme = 'deterministic'
r = 20

R_filename = 'R_files/' + 'R_' + scheme + '_' + str(r) + '.txt'

if os.path.isfile(R_filename):
    logger.info('Found existing R file, loading them!')
    R = np.loadtxt(R_filename)

else:
	logger.info('Getting row indices!')
	logger.info('Using {0}, number of rows to sample is {1}'.format( scheme, r ))
	idx = cx.comp_idx(row_p, scheme = scheme, r = r)
	logger.info('Indices are {0}!'.format(idx))

	logger.info('Getting rows!')
	R = cx.get_rows(idx)
	logger.info('Saving rows!')
	R = np.savetxt(R_filename)

logger.info('Computing relative err using R!')
relative_err = cx.comp_err(R)
logger.info('Relative error is {0}!'.format(relative_err))





