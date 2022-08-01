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
