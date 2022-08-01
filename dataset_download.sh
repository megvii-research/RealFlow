#!/bin/bash
mkdir RF_dataset
cd RF_dataset

# RF-Ktrain
mkdir RF-Ktrain
cd RF-Ktrain
wget https://data.megengine.org.cn/research/realflow/RF-Ktrain-flow.zip
cd ..

# RF-KTest
mkdir RF-KTest
cd RF-KTest
wget https://data.megengine.org.cn/research/realflow/RF-KTest-flow.zip
cd ..

# RF-Sintel
mkdir RF-Sintel
cd RF-Sintel

# RF-DAVIS
mkdir RF-DAVIS
cd RF-DAVIS
wget https://data.megengine.org.cn/research/realflow/RF-Davis-flow.zip
cd ..

# RF-AB
mkdir RF-AB
cd RF-AB
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart1.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart2.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Apart3.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart0.zip
wget https://data.megengine.org.cn/research/realflow/RFAB-flow-Bpart1.zip
cd ..


#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/models.zip s3://engine-data/research/realflow/models.zip

#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/RF-Davis/RF-Davis-flow.zip s3://engine-data/research/realflow/RF-Davis-flow.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/ALOV/part0/RFAB-flow-Apart0.zip s3://engine-data/research/realflow/RFAB-flow-Apart0.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/ALOV/part1/RFAB-flow-Apart1.zip s3://engine-data/research/realflow/RFAB-flow-Apart1.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/ALOV/part2/RFAB-flow-Apart2.zip s3://engine-data/research/realflow/RFAB-flow-Apart2.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/ALOV/part3/RFAB-flow-Apart3.zip s3://engine-data/research/realflow/RFAB-flow-Apart3.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/BDD100K/part0/RFAB-flow-Bpart0.zip s3://engine-data/research/realflow/RFAB-flow-Bpart0.zip
#aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/BDD100K/part1/RFAB-flow-Bpart1.zip s3://engine-data/research/realflow/RFAB-flow-Bpart1.zip
aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/BDD100K/part2/RFAB-flow-Bpart2.zip s3://engine-data/research/realflow/RFAB-flow-Bpart2.zip
aws s3 --endpoint-url=http://oss-cn-beijing.aliyuncs.com --profile=aliyun cp /data/ECCV/BDD100K/part3/RFAB-flow-Bpart3.zip s3://engine-data/research/realflow/RFAB-flow-Bpart3.zip