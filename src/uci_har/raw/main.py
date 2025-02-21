import toml
from data_process import X_train, y_train, X_test, y_test  # get dataset

from lib.excuter import Excuter
from models import MLP, Conv1d_3x_3, Conv2d_3x3_3, Conv2d_3x3_1, Conv1d_3x_1, BasicLSTM, BasicGRU, BiGRU, BiLSTM
from clfs import MLPClf, Conv2dClf, Conv1dClf, RNNClf

configs = toml.load('./hyper.toml')

clfs = {
    'mlp': MLPClf(**configs['mlp']['hyper'], model=MLP(
        **configs['mlp']['model'],
        input_dim=X_train.shape[1] * X_train.shape[2],
        output_dim=6,
    )),

    'conv2d_3x3_3': Conv2dClf(**configs['conv2d_3x3_3']['hyper'], model=Conv2d_3x3_3(
        **configs['conv2d_3x3_3']['model'],
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
    )),

    'conv2d_3x3_1': Conv2dClf(**configs['conv2d_3x3_1']['hyper'], model=Conv2d_3x3_1(
        **configs['conv2d_3x3_1']['model'],
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
    )),

    'conv1d_3_3': Conv1dClf(**configs['conv1d_3_3']['hyper'], model=Conv1d_3x_3(
        **configs['conv1d_3_3']['model'],
        input_width=X_train.shape[2],
        input_channels=X_train.shape[1],
        output_dim=6,
    )),

    'conv1d_3_1': Conv1dClf(**configs['conv1d_3_1']['hyper'], model=Conv1d_3x_1(
        **configs['conv1d_3_1']['model'],
        input_width=X_train.shape[2],
        input_channels=X_train.shape[1],
        output_dim=6,
    )),

    'lstm': RNNClf(**configs['lstm']['hyper'], model=BasicLSTM(
        **configs['lstm']['model'],
        input_dim=X_train.shape[1],
        output_dim=6,
    )),

    'gru': RNNClf(**configs['gru']['hyper'], model=BasicGRU(
        **configs['gru']['model'],
        input_dim=X_train.shape[1],
        output_dim=6,
    )),

    'bilstm': RNNClf(**configs['bilstm']['hyper'], model=BiLSTM(
        **configs['bilstm']['model'],
        input_dim=X_train.shape[1],
        output_dim=6,
    )),

    'bigru': RNNClf(**configs['bigru']['hyper'], model=BiGRU(
        **configs['bigru']['model'],
        input_dim=X_train.shape[1],
        output_dim=6,
    )),
}


# if you use generated config, please use this
# clfs = {
#     'mlp': MLPClf(**configs['mlp']['hyper'], model=MLP(**configs['mlp']['model'])),
#     'conv2d_3x3_3': Conv2dClf(**configs['conv2d_3x3_3']['hyper'], model=Conv2d_3x3_3(**configs['conv2d_3x3_3']['model'])),
#     'conv2d_3x3_1': Conv2dClf(**configs['conv2d_3x3_1']['hyper'], model=Conv2d_3x3_1(**configs['conv2d_3x3_1']['model'])),
#     'conv1d_3_3': Conv1dClf(**configs['conv1d_3_3']['hyper'], model=Conv1d_3x_3(**configs['conv1d_3_3']['model'])),
#     'conv1d_3_1': Conv1dClf(**configs['conv1d_3_1']['hyper'], model=Conv1d_3x_1(**configs['conv1d_3_1']['model'])),
#     'lstm': RNNClf(**configs['lstm']['hyper'], model=BasicLSTM(**configs['lstm']['model'])),
#     'gru': RNNClf(**configs['gru']['hyper'], model=BasicGRU(**configs['gru']['model'])),
#     'bilstm': RNNClf(**configs['bilstm']['hyper'], model=BiLSTM(**configs['bilstm']['model'])),
#     'bigru': RNNClf(**configs['bigru']['hyper'], model=BiGRU(**configs['bigru']['model'])),
# }

exc = Excuter(X_train, y_train, X_test, y_test,
              clf_dict=clfs,
              log=True,
              log_dir='./log/')
exc.run_all()
