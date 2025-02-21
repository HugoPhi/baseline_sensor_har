import time
import torch as tc
import joblib
from lib.clfs import Clfs


class MLClfs(Clfs):
    def __init__(self, model, param_file=None):
        self.model = model
        self.param_file = 'model_filename.pkl'
        self.training_time = -1
        self.testing_time = -1

    def fit(self, X_train, y_train, load=False):
        if load:
            self.model = joblib.load('model_filename.pkl')

        start = time.time()
        self.model.fit(X_train, y_train.ravel())
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        start = time.time()
        y_pred = self.model.predict(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze()

    def predict_proba(self, x_test):
        start = time.time()
        y_pred = self.model.predict_proba(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred

    def get_params(self):
        hyper = dict()
        model = self.model.get_params()
        model = dict(filter(lambda item: not item[0].startswith('base_estimator'), model.items()))  # filt the base_estimator by AdaBoost
        return hyper, model

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class XGBClfs(Clfs):
    def __init__(self, model, param_file=None):
        self.model = model
        self.param_file = 'model_filename.pkl'
        self.training_time = -1
        self.testing_time = -1

    def fit(self, X_train, y_train, load=False):
        if load:
            self.model = joblib.load('model_filename.pkl')

        start = time.time()
        self.model.fit(X_train, y_train.ravel() - 1)
        end = time.time()
        self.training_time = end - start

    def predict(self, x_test):
        start = time.time()
        y_pred = self.model.predict(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze() + 1

    def predict_proba(self, x_test):
        start = time.time()
        y_pred = self.model.predict_proba(x_test)
        end = time.time()
        self.testing_time = end - start
        return y_pred

    def get_params(self):
        hyper = dict()
        model = self.model.get_params()
        return hyper, model

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time


class MLPClf(Clfs):
    def __init__(self, epochs, lr, model):
        self.params = {key: value for key, value in locals().items() if key != 'self' and key != 'model'}  # 这里保证可以复刻这个Clf

        self.epochs = epochs
        self.lr = lr
        self.model = model

        # 定义损失函数和优化器
        self.criterion = tc.nn.CrossEntropyLoss()
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train, load=False):
        # 训练循环
        X_train_tensor = tc.tensor(X_train, dtype=tc.float32)
        y_train_tensor = tc.tensor(y_train - 1, dtype=tc.long).reshape(-1)
        start = time.time()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        end = time.time()
        self.training_time = end - start

    def predict_proba(self, x_test):
        # 测试阶段
        start = time.time()
        self.model.eval()
        with tc.no_grad():
            X_test_tensor = tc.tensor(x_test, dtype=tc.float32)
            test_outputs = self.model(X_test_tensor)
            test_outputs = tc.nn.functional.softmax(test_outputs, dim=1)
        end = time.time()
        self.testing_time = end - start
        return test_outputs.numpy()

    def predict(self, x_test):
        # 测试阶段
        start = time.time()
        self.model.eval()
        with tc.no_grad():
            X_test_tensor = tc.tensor(x_test, dtype=tc.float32)
            test_outputs = self.model(X_test_tensor)
            y_pred = test_outputs.argmax(dim=1).numpy()
        end = time.time()
        self.testing_time = end - start
        return y_pred.squeeze() + 1  # class from [1, 2, 3, 4, 5, 6]

    def get_params(self):
        hyper = self.params
        model = dict()
        return hyper, model

    def get_training_time(self):
        return self.training_time

    def get_testing_time(self):
        return self.testing_time
