#!/bin/bash

# 定义下载链接和目标目录
URL="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
DATA_DIR="data"

# 创建目标目录（如果不存在）
if [ ! -d "$DATA_DIR" ]; then
  mkdir "$DATA_DIR"
fi

# 下载数据集
echo "Downloading dataset from $URL..."
# curl -L $URL -o "$DATA_DIR/wisdm_dataset.zip"

# 检查下载是否成功
if [ $? -eq 0 ]; then
  echo "Dataset downloaded successfully to $DATA_DIR/wisdm_dataset.zip"

  # 解压文件
  echo "Extracting files..."
  unzip "$DATA_DIR/wisdm_dataset.zip" -d "$DATA_DIR"
  echo "Dataset extracted to $DATA_DIR"
else
  echo "Download failed."
fi
