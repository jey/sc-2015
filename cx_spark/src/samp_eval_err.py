from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel 
from utils import prepare_matrix
from cx import CX
from sparse_row_matrix import SparseRowMatrix

conf = SparkConf().set('spark.eventLog.enabled','true')
#.set('spark.driver.maxResultSize', '15g').set("spark.executor.memory", "30g")
sc = SparkContext(appName='samp_eval_err',conf=conf)
import ast
import numpy as np
import logging.config
import logging

logging.config.fileConfig('logging.cfg', disable_existing_loggers=False) 
logger = logging.getLogger(__name__)

def parse(string):
    s = str(string)
    val = ast.literal_eval(s)
    return val[0], (np.array(val[1][0]), np.array(val[1][1]))

data = sc.textFile('msi_data',500).map(lambda x:parse(x))

#row_shape = 131048
#column_shape = 8258911
#131047 8258910
row_shape = 8258911
column_shape = 131048
#column_shape+=20

#print data.take(1)
#print data.count()

matrix_A = SparseRowMatrix(data, 'output', row_shape, column_shape, False)

#row_lev = np.loadtxt( '../results/row_leverage_scores_logged' )
row_p = np.loadtxt( '../results/row_p_scores_logged' )
print row_p.shape

cx = CX(matrix_A)

#idx = cx.comp_idx(row_p, scheme='deterministic', r=50)
#print idx

R = np.loadtxt('temp_R.txt')

sum_norms = cx.comp_err(R)

print sum_norms



