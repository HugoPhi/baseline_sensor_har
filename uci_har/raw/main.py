import toml
from data_process import X_train, y_train, X_test, y_test  # get dataset
from models import MLP, Conv1d_3x_3, Conv2d_3x3_3, Conv2d_3x3_1, BasicRNN
from clfs import MLPClf, Conv2dClf, Conv1dClf, RNNClf

configs = toml.load("hyper.toml")


# clf name : Clf Template(Model)
clfs = {
    "mlp 4hr": MLPClf(X_train, y_train, configs["mlp 4hr"], train_log=configs["mlp 4hr"]["train_log"], model=MLP(
        input_dim=X_train.shape[1] * X_train.shape[2],
        hidden_dims=[256, 128, 64, 32],
        output_dim=6,
        dropout=configs["mlp 4hr"]["dropout"],
    )),

    "conv2d_3x3 3hr": Conv2dClf(X_train, y_train, configs["conv2d_3x3 3hr"], train_log=configs["conv2d_3x3 3hr"]["train_log"], model=Conv2d_3x3_3(
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
        dropout=configs["conv2d_3x3 3hr"]["dropout"],
    )),

    "conv2d_3x3 1hr": Conv2dClf(X_train, y_train, configs["conv2d_3x3 1hr"], train_log=configs["conv2d_3x3 1hr"]["train_log"], model=Conv2d_3x3_1(
        input_height=X_train.shape[1],
        input_width=X_train.shape[2],
        input_channels=1,
        output_dim=6,
        dropout=configs["conv2d_3x3 3hr"]["dropout"],
    )),

    "conv1d_3 3hr": Conv1dClf(X_train, y_train, configs["conv1d_3 3hr"], train_log=configs["conv1d_3 3hr"]["train_log"], model=Conv1d_3x_3(
        input_width=X_train.shape[2],
        input_channels=X_train.shape[1],
        output_dim=6,
        dropout=configs["conv1d_3 3hr"]["dropout"],
    )),

    "rnn 1hr": RNNClf(X_train, y_train, configs["rnn 1hr"], train_log=configs["rnn 1hr"]["train_log"], model=BasicRNN(
        input_dim=X_train.shape[1],
        hidden_dim=configs["rnn 1hr"]["hidden_dims"],
        output_dim=6,
        dropout=configs["rnn 1hr"]["dropout"]
    ))
}

for name, clf in clfs.items():
    print(f">> {name}: ", end='')
    clf.fit()
    print(f'{clf.training_time} s')
    y_pred = clf.predict(X_test)
    y_pred = y_pred.detach().numpy()
    acc = (y_pred == (y_test - 1)).mean()
    print(acc * 100)
