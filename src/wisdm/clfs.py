import yaml
import torch
from torch import nn
import time
from plugins.lrkit.clfs import Clfs, timing

from data_process import X_train  # get dataset
from models import mlp, Conv1d_3x_3, Conv2d_3x3_3, Conv2d_3x3_1, Conv1d_3x_1, BasicLSTM, BasicGRU, BiGRU, BiLSTM


def astag(cls):
    tag = f'!{cls.__name__}'

    def constructor(loader, node):
        params = loader.construct_mapping(node)
        return cls(**params)

    yaml.add_constructor(tag, constructor)
    return cls


class NNClfs(Clfs):
    '''
    NNClfs(Clfs): 相比与Clfs，加入了xpip，用于将数据集转换成模型可接受的输入。

    ## Methods:
      - ... (从Clfs继承的部分)
      - xpip(self, x): 将x进行处理，返回x

    ## Params:
      - lr: 学习率
      - epochs: 训练轮数
      - model: 模型实例
      - train_log(bool=False): 是否显示训练过程
      - pre_trained(str='./model.m5'): 训练好的模型的文件路径
    '''

    def __init__(self, lr, epochs, batch_size, train_log=False):
        super(NNClfs, self).__init__()

        self.train_log = train_log
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None

        self.criterion = None
        self.optimizer = None

    def xpip(self, x):
        '''
        这个函数的作用是将输入数据转换成合适的形状
        '''
        return x

    def fit(self, X_train, y_train, load=False):
        if not load:
            dataset = torch.utils.data.TensorDataset(self.xpip(X_train), y_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            start_time = time.time()
            if self.train_log:
                print()
            for epoch in range(self.epochs):
                for i, (x, y) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(x.float())
                    loss = self.criterion(outputs, torch.argmax(y, dim=1))
                    loss.backward()
                    self.optimizer.step()

                if self.train_log:
                    print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}, Train Acc: {100 * (self.predict(X_train) == torch.argmax(y_train, dim=1)).float().mean().item():.2f}%')
            self.training_time = time.time() - start_time
        else:
            self.model.load_state_dict(torch.load(self.pre_trained))
            print('Pre-trained model loaded.')

    @timing
    def predict(self, X_test):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.xpip(X_test).float())
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    @timing
    def predict_proba(self, X_test):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.xpip(X_test).float())
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs.numpy()


@astag
class MLPClf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 hidden_dims,
                 dropout=0.2,
                 train_log=False):
        super(MLPClf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = mlp(
            input_dim=X_train.shape[1] * X_train.shape[2],
            hidden_dims=hidden_dims,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return x.reshape(x.shape[0], -1)


@astag
class Conv2d_3x3_1_Clf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 dropout=0.2,
                 train_log=False):
        super(Conv2d_3x3_1_Clf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = Conv2d_3x3_1(
            input_height=X_train.shape[1],
            input_width=X_train.shape[2],
            input_channels=1,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])


@astag
class Conv2d_3x3_3_Clf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 dropout=0.2,
                 train_log=False):
        super(Conv2d_3x3_3_Clf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = Conv2d_3x3_3(
            input_height=X_train.shape[1],
            input_width=X_train.shape[2],
            input_channels=1,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])


@astag
class Conv1d_3x_1_Clf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 dropout=0.2,
                 train_log=False):
        super(Conv1d_3x_1_Clf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = Conv1d_3x_1(
            input_width=X_train.shape[2],
            input_channels=X_train.shape[1],
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


@astag
class Conv1d_3x_3_Clf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 dropout=0.2,
                 train_log=False):
        super(Conv1d_3x_3_Clf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = Conv1d_3x_3(
            input_width=X_train.shape[2],
            input_channels=X_train.shape[1],
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


@astag
class LSTMClf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 hidden_dim, num_layers,
                 dropout=0.2,
                 train_log=False):
        super(LSTMClf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = BasicLSTM(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)


@astag
class GRUClf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 hidden_dim, num_layers,
                 dropout=0.2,
                 train_log=False):
        super(GRUClf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = BasicGRU(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)


@astag
class BiLSTMClf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 hidden_dim, num_layers,
                 dropout=0.2,
                 train_log=False):
        super(BiLSTMClf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = BiLSTM(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)


@astag
class BiGRUClf(NNClfs):
    def __init__(self, lr, epochs, batch_size,
                 hidden_dim, num_layers,
                 dropout=0.2,
                 train_log=False):
        super(BiGRUClf, self).__init__(lr, epochs, batch_size, train_log)

        self.model = BiGRU(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=6,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)
