from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel 
from utils import prepare_matrix
from cx import CX
from sparse_row_matrix import SparseRowMatrix

conf = SparkConf().set('spark.eventLog.enabled','true')
#.set('spark.driver.maxResultSize', '15g').set("spark.executor.memory", "30g")
sc = SparkContext(appName='cx_exp',conf=conf)
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

print data.take(1)
#print data.count()

matrix_A = SparseRowMatrix(data, 'output', row_shape, column_shape, False)
cx = CX(matrix_A)
k = 15
q = 4
lev_row, lev_col, p_row, p_col = cx.get_lev(k=k, q=q) 
#end = time.time()
row_leverage_scores_file='row_leverage_scores_logged'
row_p_scores_file='row_p_scores_logged'
col_leverage_scores_file='col_leverage_scores_logged'
col_p_scores_file='col_p_scores_logged'
np.savetxt(row_leverage_scores_file, np.array(lev_row))
np.savetxt(row_p_scores_file, np.array(p_row))
np.savetxt(col_leverage_scores_file, np.array(lev_col))
np.savetxt(col_p_scores_file, np.array(p_col))


"""
def parse_func(x):
    stringed  = str(x)
    chunks = stringed.split(",")
    return int(chunks[1]), int(chunks[0]), float(chunks[2])

data = sc.textFile("/scratch1/scratchdirs/msingh/sc_paper/experiments/striped_data/final_matrix").map(lambda x:    parse_func(x))
#rows_rdd = data.map(lambda x:str(x)).map(lambda x:x.split(',')).map(lambda x:(int(x[0]), int(x[1]), float(x[2])))
#sorted_Rdd = prepare_matrix(rows_rdd)
#sorted_Rdd.saveAsTextFile('/scratch1/scratchdirs/msingh/sc_paper/experiments/striped_data/rows_matrix')
#columns_rdd = data.map(lambda x: (x[1],x[0],x[2]))
csorted_rdd = prepare_matrix(data)
csorted_rdd.saveAsTextFile('/scratch1/scratchdirs/msingh/sc_paper/experiments/striped_data/ncolumns_matrix')
print "completed"

d = data.take(2)
r = rdd.take(2)
print d
print r
rdd.map(lambda x:str(x)).map(lambda x:x.split(',')[1]).saveAsTextFile('/scratch1/scratchdirs/msingh/sc_paper/full_output/scratch/columns1')
"""
#.map(lambda x:str(x)).map(lambda x:x.split(',')[1]).saveAsTextFile('/scratch1/scratchdirs/msingh/sc_paper/full_output/scratch/columns')

