import pandas as pd
import numpy as np
import torch as tc
import kagglehub


# Download latest version
path = kagglehub.dataset_download("drsaeedmohsen/ucihar-dataset")
path += '/UCI-HAR Dataset'
print("Path to dataset files:", path)

TRAIN_PATH = path + '/train/Inertial Signals/'
TEST_PATH = path + '/test/Inertial Signals/'

PREFIXS = [
    'body_acc_x_',
    'body_acc_y_',
    'body_acc_z_',
    'body_gyro_x_',
    'body_gyro_y_',
    'body_gyro_z_',
    'total_acc_x_',
    'total_acc_y_',
    'total_acc_z_',
]


X_train = []
for prefix in PREFIXS:
    X_train.append(pd.read_csv(TRAIN_PATH + prefix + 'train.txt', header=None, delim_whitespace=True).to_numpy())

X_train = np.transpose(np.array(X_train), (1, 0, 2))
X_train = tc.tensor(X_train)

X_test = []
for prefix in PREFIXS:
    X_test.append(pd.read_csv(TEST_PATH + prefix + 'test.txt', header=None, delim_whitespace=True).to_numpy())
X_test = np.transpose(np.array(X_test), (1, 0, 2))
X_test = tc.tensor(X_test)


y_train = pd.read_csv(path + '/train/y_train.txt', header=None).to_numpy().squeeze()
y_test = pd.read_csv(path + '/test/y_test.txt', header=None).to_numpy().squeeze() - 1

# 将标签转换为 one-hot 编码
y_train = tc.nn.functional.one_hot(tc.tensor(y_train - 1), 6)

print('X_train 形状:', X_train.shape)  # 应为 (7352, 9, 128)
print('y_train 形状:', y_train.shape)  # 应为 (7352, 6)
