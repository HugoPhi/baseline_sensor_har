import pandas as pd
import numpy as np

X_train = pd.read_csv(
    '../data/train/X_train.txt',
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_train = pd.read_csv(
    '../data/train/y_train.txt',
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

X_test = pd.read_csv(
    '../data/test/X_test.txt',
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_test = pd.read_csv(
    '../data/test/y_test.txt',
    sep=r'\s+',
    header=None,
    engine='python').to_numpy() - 1

shuffle = np.random.permutation(len(X_train))
print(f'train: {len(X_train)}, test: {len(X_test)}')
X_train, y_train = X_train[shuffle], y_train[shuffle] - 1
