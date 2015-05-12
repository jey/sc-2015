#!/bin/bash
spark-submit --driver-java-options '-Dlog4j.configuration=log4j.properties' --executor-memory 7G --driver-memory 8G \
 --py-files conf/settings.cfg,parse_config.py,sparse_row_matrix.py,spark_msi.py,cx.py,rma_utils.py,utils.py,randomized_svd.py samp_eval_err.py $@ 2>&1 | tee test.log

#spark-submit --driver-java-options '-Dlog4j.configuration=log4j.properties' --executor-memory 7G --driver-memory 8G \
# --py-files conf/settings.cfg,parse_config.py,sparse_row_matrix.py,spark_msi.py,cx.py,rma_utils.py,utils.py create_small_msi_data.py $@ 2>&1 | tee test.log
