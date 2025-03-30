import yaml
import kagglehub
import pandas as pd
import numpy as np

with open('./data.yml') as f:
    path = yaml.safe_load(f)['path']

# 下载最新的数据集版本
if path is None:
    path = kagglehub.dataset_download("drsaeedmohsen/ucihar-dataset")

data_path = f"{path}/UCI-HAR Dataset"

# 构建数据文件路径
train_x_path = f"{data_path}/train/X_train.txt"
train_y_path = f"{data_path}/train/y_train.txt"
test_x_path = f"{data_path}/test/X_test.txt"
test_y_path = f"{data_path}/test/y_test.txt"

# 加载训练和测试数据
X_train = pd.read_csv(train_x_path, sep=r'\s+', header=None, engine='python').to_numpy()
y_train = pd.read_csv(train_y_path, sep=r'\s+', header=None, engine='python').to_numpy()
X_test = pd.read_csv(test_x_path, sep=r'\s+', header=None, engine='python').to_numpy()
y_test = pd.read_csv(test_y_path, sep=r'\s+', header=None, engine='python').to_numpy() - 1

# 打乱训练数据顺序
shuffle = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle], y_train[shuffle] - 1

print(f"训练样本数量: {len(X_train)}, 测试样本数量: {len(X_test)}")
