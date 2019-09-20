#!/bin/bash
# 创建新环境，环境名为fastertf3
conda create -n fastertf python=3.6
source activate fastertf
# 安装python库，gpu则要安装tensorflow-gpu版本，与CUDA和显卡相对应的版本
pip install cmake -i https://pypi.douban.com/simple
pip install tensorflow-gpu==1.13.1 -i https://pypi.douban.com/simple
# 使用tensorflow cpu版本
# pip install Tensorflow==1.13.1 -i https://pypi.douban.com/simple
# 创建并编译fastertf的环境
mkdir -p build
cd build
# 导入安装好的tensorflow
cmake -DSM=60 -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/home/jovenchu/anaconda3/envs/fastertf3/lib/python3.6/site-packages/tensorflow .. # Tensorflow mode
make
# 修改模型训练参数
./bin/gemm_fp32 16 128 12 64