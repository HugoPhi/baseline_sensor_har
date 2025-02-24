import toml
from data_process import X_train, y_train, X_test, y_test  # get dataset

from plugins.executer import Executer
from clfs import MLPClf, Conv1d_3x_1_Clf, Conv1d_3x_3_Clf, Conv2d_3x3_1_Clf, Conv2d_3x3_3_Clf, LSTMClf, GRUClf, BiLSTMClf, BiGRUClf

configs = toml.load('./hyper.toml')

# if you use generated config, please use this
clfs = {
    'mlp': MLPClf(**configs['mlp']),
    'conv2d_3x3_3': Conv2d_3x3_3_Clf(**configs['conv2d_3x3_3']),
    'conv2d_3x3_1': Conv2d_3x3_1_Clf(**configs['conv2d_3x3_1']),
    'conv1d_3_3': Conv1d_3x_3_Clf(**configs['conv1d_3_3']),
    'conv1d_3_1': Conv1d_3x_1_Clf(**configs['conv1d_3_1']),
    'lstm': LSTMClf(**configs['lstm']),
    'gru': GRUClf(**configs['gru']),
    'bilstm': BiLSTMClf(**configs['bilstm']),
    'bigru': BiGRUClf(**configs['bigru']),
}

exc = Executer(X_train, y_train, X_test, y_test,
               clf_dict=clfs,
               log=True,
               log_dir='./log/')
exc.run_all(sort_by='accuracy', ascending=False)
