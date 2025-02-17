from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import pandas as pd

PATH = '../data/train/Inertial Signals/'
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
    X_train.append(pd.read_csv(PATH + prefix + 'train.txt', header=None, delim_whitespace=True).to_numpy())

X_train = np.transpose(np.array(X_train), (1, 0, 2))

y_train = pd.read_csv('../data/train/y_train.txt', header=None).to_numpy().squeeze()

# Conv1d
# 将标签转换为 one-hot 编码
num_classes = 6
y_train = to_categorical(y_train - 1, num_classes=num_classes)

# 检查形状
print("X_train 形状:", X_train.shape)  # 应为 (7352, 9, 128)
print("y_train 形状:", y_train.shape)  # 应为 (7352, 6)


# MLP
print()
