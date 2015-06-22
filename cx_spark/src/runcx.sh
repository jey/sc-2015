spark-submit --verbose \
  --driver-memory 96G \
  --conf spark.driver.maxResultSize=64g \
  $1 \
  csv hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat.csv \
  8258911 131048 \
  smdump-24-4-4-orth.out \
  24 4 4

#  genmat \
#  hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat.csv \
#  hdfs:///test.out

#  df hdfs:///test.out \
#  8258911 131048 \
#  dump.out \
#  16 2 1


#  genmat hdfs:///Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-smoothed-mz=437.11407-sd=0.05.rawmat.csv.gz \
#  hdfs:///smoothed.out

#  genmat s3n://amp-jey/sc-2015/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat/Lewis_Dalisay_Peltatum_20131115_hexandrum_1_1-masked.mat.csv \
