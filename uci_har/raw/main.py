import torch as tc
import numpy as np
import pandas as pd
from clfs import MLPClf, Conv2d1cClf, Conv1dxcClf

TRAIN_PATH = '../data/train/Inertial Signals/'
TEST_PATH = '../data/test/Inertial Signals/'
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


y_train = pd.read_csv('../data/train/y_train.txt', header=None).to_numpy().squeeze()
y_test = pd.read_csv('../data/test/y_test.txt', header=None).to_numpy().squeeze()

# Conv1d
# 将标签转换为 one-hot 编码
y_train = tc.nn.functional.one_hot(tc.tensor(y_train - 1), 6)

# 检查形状
print("X_train 形状:", X_train.shape)  # 应为 (7352, 9, 128)
print("y_train 形状:", y_train.shape)  # 应为 (7352, 6)


# MLP
if False:
    config = {
        "input_dim": X_train.shape[1] * X_train.shape[2],
        "hidden_dims": [256, 128, 64, 32],
        "output_dim": 6,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 32
    }
    mlp_clf = MLPClf(X_train, y_train, config)
    mlp_clf.fit()

    # 预测
    y_pred = mlp_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    for k, v in mlp_clf.hyper_info().items():
        print(f" * {k}: {v}")


# Conv2d with 1 channel
if False:
    config = {
        "output_dim": 6,
        "dropout": 0.5,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "layers_config": [
            {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 32, "use_bn": False, "padding": 1},
            # {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 64, "use_bn": True, "padding": 1},
            # {"kernel_size": (3, 3), "pool_size": (2, 2), "filters": 128, "use_bn": False, "padding": 1}
        ]
    }

    cnn_clf = Conv2d1cClf(X_train, y_train, config)
    cnn_clf.fit()

    # 预测
    y_pred = cnn_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    for k, v in cnn_clf.hyper_info().items():
        print(f" * {k}: {v}")

# Conv1d with 9 channel
if False:
    config = {
        "output_dim": 6,
        "dropout": 0.5,
        "lr": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "layers_config": [
            {"kernel_size": (3,), "pool_size": (2,), "filters": 32, "use_bn": True},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 64, "use_bn": True},
            {"kernel_size": (3,), "pool_size": (2,), "filters": 128, "use_bn": False}
        ]
    }

    cnn_clf = Conv1dxcClf(X_train, y_train, config)
    cnn_clf.fit()

    # 预测
    y_pred = cnn_clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).float().mean().item()
    print(acc)
    for k, v in cnn_clf.hyper_info().items():
        print(f" * {k}: {v}")
