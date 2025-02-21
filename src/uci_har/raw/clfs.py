import torch
from torch import nn
import time
from lib.clfs import Clfs


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

    def __init__(self, lr, epochs, batch_size, model, train_log=False, pre_trained='./model.m5'):
        self.train_log = train_log
        self.params = {key: value for key, value in locals().items() if key != 'self' and key != 'model'}  # 这里保证可以复刻这个Clf

        self.epochs = epochs
        self.batch_size = batch_size

        self.pre_trained = pre_trained

        self.training_time = -1
        self.testing_time = -1

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
                    print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}, Train Acc: {100 * (self.predict(X_train) == torch.argmax(y_train, dim=1)).float().mean().item():.2f}%')
            self.training_time = time.time() - start_time
        else:
            self.model.load_state_dict(torch.load(self.pre_trained))
            print('Pre-trained model loaded.')

    def predict(self, X_test):
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.xpip(X_test).float())
            _, predicted = torch.max(outputs.data, 1)
        self.testing_time = time.time() - start_time
        return predicted.numpy()

    def predict_proba(self, X_test):
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.xpip(X_test).float())
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        self.testing_time = time.time() - start_time
        return outputs.numpy()

    def get_params(self):
        hyper = self.params
        model = self.model.out_params
        return hyper, model

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class MLPClf(NNClfs):
    def __init__(self, *args, **dargs):
        super(MLPClf, self).__init__(*args, **dargs)

    def xpip(self, x):
        return x.reshape(x.shape[0], -1)


class Conv2dClf(NNClfs):
    def __init__(self, *args, **dargs):
        super(Conv2dClf, self).__init__(*args, **dargs)

    def xpip(self, x):
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])


class Conv1dClf(NNClfs):
    def __init__(self, *args, **dargs):
        super(Conv1dClf, self).__init__(*args, **dargs)

    def xpip(self, x):
        return x


class RNNClf(NNClfs):
    def __init__(self, *args, **dargs):
        super(RNNClf, self).__init__(*args, **dargs)

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)
