from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.storagelevel import StorageLevel
from rma_utils import form_csr_matrix

import ast
import numpy as np


conf = SparkConf().set('spark.eventLog.enabled','true')
#.set('spark.driver.maxResultSize', '15g').set("spark.executor.memory", "30g")
sc = SparkContext(appName='small_msi_data',conf=conf)

def parse(string):
    s = str(string)
    val = ast.literal_eval(s)
    return val[0], (np.array(val[1][0]), np.array(val[1][1]))

def keep(row, p, col_idx):
    coin = np.random.rand()
    col = []
    elem = []

    if coin < p:
        for i, j in zip( row[1][0], row[1][1] ) :
            if i in col_idx:
                col.append(np.where(col_idx==i)[0][0])
                elem.append(j)
        return [col, elem]

m = 8258911
n = 131048

data = sc.textFile('msi_data',500).map(lambda x:parse(x))
col_idx = np.random.permutation(n)[:2e4]

small_data_sparse = data.map(lambda row: keep(row, 2e4/m, col_idx)).filter(lambda row: row is not None).collect()
#small_data_sparse = data.take(5) # sparse format: [row_idx, [col_idx1, col_idx2, ...], [elem1, elem2, ...]]

data = {'row':[],'col':[],'val':[]}
row_idx = 0
for r in small_data_sparse:
    data['row'] += [row_idx]*len(r[0])
    data['col'] += r[0]
    data['val'] += r[1]
    row_idx += 1

data = form_csr_matrix(data,row_idx,2e4).toarray()

m, n = data.shape
print data.shape

#msi_sub = data[:,np.random.permutation(n)[:2e4]]
#print msi_sub.shape

np.savetxt('msi_sub.txt', data)

