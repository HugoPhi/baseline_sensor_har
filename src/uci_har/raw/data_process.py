import yaml
import pandas as pd
import numpy as np
import kagglehub


def one_hot(x, C):
    res = np.zeros((x.shape[0], C))
    res[np.arange(x.shape[0]), x] = 1
    return res


with open('./data.yml') as f:
    path = yaml.safe_load(f)['path']

# Download latest version
if path is None:
    path = kagglehub.dataset_download("drsaeedmohsen/ucihar-dataset") + '/UCI-HAR Dataset'
else:
    path = path + '/UCI-HAR Dataset'
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
X_train = np.transpose(np.array(X_train), (1, 0, 2)).astype(np.float32)  # Convert into float32

X_test = []
for prefix in PREFIXS:
    X_test.append(pd.read_csv(TEST_PATH + prefix + 'test.txt', header=None, delim_whitespace=True).to_numpy())
X_test = np.transpose(np.array(X_test), (1, 0, 2)).astype(np.float32)


y_train = pd.read_csv(path + '/train/y_train.txt', header=None).to_numpy().squeeze()
y_test = pd.read_csv(path + '/test/y_test.txt', header=None).to_numpy().squeeze() - 1

# one hot & to torch
y_train = one_hot(y_train - 1, 6)

# shuffle
shuffle = np.random.permutation(X_train.shape[0])
X_train = X_train[shuffle]
y_train = y_train[shuffle]

print('X_train 形状:', X_train.shape)  # (7352, 9, 128)
print('y_train 形状:', y_train.shape)  # (7352, 6)
print('X_train 形状:', X_test.shape)  # (7352, 9, 128)
print('y_train 形状:', y_test.shape)  # (7352, 6)
