import toml
from data_process import X_train, y_train, X_test, y_test  # get dataset
from models import MLP, Conv1d_3x_3, Conv2d_3x3_3, Conv2d_3x3_1, Conv1d_3x_1, BasicLSTM
# from models import BasicRNN
from clfs import MLPClf, Conv2dClf, Conv1dClf, RNNClf

configs = toml.load('hyper.toml')


clfs = {
    'mlp 4hr': MLPClf(**configs['mlp 4hr'], model=MLP(
        input_dim=X_train.shape[1] * X_train.shape[2],
        hidden_dims=[256, 128, 64, 32],
        output_dim=6,
        dropout=configs['mlp 4hr']['dropout'],
    )),

    'conv2d_3x3 3hr': Conv2dClf(**configs['conv2d_3x3 3hr'], model=Conv2d_3x3_3(
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
        dropout=configs['conv2d_3x3 3hr']['dropout'],
    )),

    'conv2d_3x3 1hr': Conv2dClf(**configs['conv2d_3x3 1hr'], model=Conv2d_3x3_1(
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
        dropout=configs['conv2d_3x3 3hr']['dropout'],
    )),

    'conv1d_3 3hr': Conv1dClf(**configs['conv1d_3 3hr'], model=Conv1d_3x_3(
        input_width=X_train.shape[2],
        input_channels=X_train.shape[1],
        output_dim=6,
        dropout=configs['conv1d_3 3hr']['dropout'],
    )),

    'conv1d_3 1hr': Conv1dClf(**configs['conv1d_3 1hr'], model=Conv1d_3x_1(
        input_width=X_train.shape[2],
        input_channels=X_train.shape[1],
        output_dim=6,
        dropout=configs['conv1d_3 1hr']['dropout'],
    )),

    'lstm 1hr': RNNClf(**configs['lstm 1hr'], model=BasicLSTM(
        input_dim=X_train.shape[1],
        hidden_dim=configs['lstm 1hr']['hidden_dims'],
        output_dim=6,
        dropout=configs['lstm 1hr']['dropout']
    ))
}

for name, clf in clfs.items():
    print(f'>> {name}: ', end='')
    clf.fit(X_train, y_train)
    print(f'{clf.training_time} s')
    y_pred = clf.predict(X_test)
    acc = (y_pred == (y_test - 1)).mean()
    print(acc * 100)
