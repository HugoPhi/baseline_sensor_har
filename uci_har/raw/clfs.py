import torch
from torch import nn
from abc import ABC, abstractmethod
import time


class Clfs(ABC):
    '''
    Classifier 接口：
      - fit(self, load=Flase): 训练分类器
      - predict(self, x): 返回x的预测值
      - hyper_info(self): 返回超参数字典
      - get_training_time(self): 获取训练时间，如果load=False
      - get_testing_time(self): 获取测试时间
    '''
    # 1. 快速取用
    # 2. 参数信息获取
    # 3. 统一API接口

    @abstractmethod
    def predict(self):
        '''
        !!! 在这里要统计预测时间
        '''
        pass

    @abstractmethod
    def fit(self, load=False):
        '''
        !!! 在这里要统计测试时间
        '''
        pass

    @abstractmethod
    def hyper_info(self) -> dict:
        pass

    @abstractmethod
    def get_training_time(self):
        pass

    @abstractmethod
    def get_testing_time(self):
        pass


class NNClfs(Clfs):
    def __init__(self, X, y, config, model, train_log=False, pre_trained="./model.m5"):
        self.train_log = train_log
        self.config = config
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.pre_trained = pre_trained

        self.X = X
        self.y = y

        self.training_time = -1
        self.testing_time = -1

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def xpip(self, x):
        '''
        这个函数的作用是将输入数据转换成合适的形状
        '''
        return x

    def fit(self, load=False):
        if not load:
            X_train = self.xpip(self.X)
            y_train = self.y
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
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
                    print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}, Train Acc: {100 * (self.predict(self.X) == torch.argmax(self.y, dim=1)).float().mean().item():.2f}%')
            self.training_time = time.time() - start_time
        else:
            print("Loading pre-trained model...")
            self.model.load_state_dict(torch.load(self.pre_trained))
            print("Pre-trained model loaded.")

    def predict(self, X_test):
        start_time = time.time()
        X_test = self.xpip(X_test)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_test.float())
            _, predicted = torch.max(outputs.data, 1)
        self.testing_time = time.time() - start_time
        return predicted

    def hyper_info(self) -> dict:
        return self.config

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
