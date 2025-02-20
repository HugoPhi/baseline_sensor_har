import torch
from torch import nn
from abc import ABC, abstractmethod
import time
import numpy as np


class Clfs(ABC):
    '''
    Classifier 接口：
      - fit(self, X_train, y_train, load=Flase): 训练分类器
      - predict(self, x_test): 返回x_test的预测值
      - predict_proba(self, x_test): 返回x_test的预测概率矩阵
      - hyper_info(self): 返回超参数字典
      - get_training_time(self): 获取训练时间，如果load=False
      - get_testing_time(self): 获取测试时间

    * 这样的接口设计可以使得学习器的训练和学习过程有一个同一的API，便于不同框架下模型的对比
    * 新版的这套接口让"预测器"和将要使用的数据集分割开；让预测器和模型的具体架构分割开
    '''
    @abstractmethod
    def __init__(self, model):
        '''
        - 这里面需要包含配置文件里面的所有参数，用于配置self.config，便于调试
        '''
        self.config = dict()
        self.model = model
        self.training_time = -1
        self.testing_time = -1

    @abstractmethod
    def predict(self, x_test) -> np.ndarray:
        '''
        - 这里返回和y_train同样形式的结果向量
        - 在这里要统计预测时间
        '''
        pass

    @abstractmethod
    def predict_proba(self, x_test) -> np.ndarray:
        '''
        - 这里返回经过softmax之后的概率矩阵
        '''
        pass

    @abstractmethod
    def fit(self, X_train, y_train, load=False):
        '''
        - 在这里要统计测试时间
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
      - pre_trained(str="./model.m5"): 训练好的模型的文件路径
    '''

    def __init__(self, lr, epochs, batch_size, dropout, model, train_log=False, pre_trained="./model.m5"):
        self.train_log = train_log
        self.config = {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout": dropout,
            "model arch": model,
        }
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.pre_trained = pre_trained

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
            print("Pre-trained model loaded.")

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
            outputs = torch.nn.functional.softmax(outputs)
        self.testing_time = time.time() - start_time
        return outputs.numpy()

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
    def __init__(self, hidden_dims, *args, **dargs):
        super(RNNClf, self).__init__(*args, **dargs)

        self.config["hidden_dims"] = hidden_dims

    def xpip(self, x):
        return torch.transpose(x, 1, 2)  # 形状是 (batch_size, seq_len, input_dim)
