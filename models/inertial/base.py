import yaml
import torch
from plugins.lrkit import ClfTrait, timing

with open('./data.yml', 'r') as f:
    config = yaml.safe_load(f)

'''
F: features
T: time steps
C: classes
N_train: number of training samples
N_test: number of test samples
'''
F, T, C, N_train, N_test = config['F'], config['T'], config['C'], config['N_train'], config['N_test']


def astag(cls):
    '''
    register class as a tag for yaml, with format: "!{$class_name}", for example, 'class CNN: ...' -> '!CNN' in yaml.
    '''

    tag = f'!{cls.__name__}'

    def constructor(loader, node):
        params = loader.construct_mapping(node)
        return cls(**params)

    yaml.add_constructor(tag, constructor)
    return cls


class nnclassifier(ClfTrait):
    def __init__(self, lr, epochs, batch_size):
        super(nnclassifier, self).__init__()

        self.epochs = epochs
        self.batch_size = batch_size

    def flash_training(self):
        pass

    def xpip(self, x):
        '''
        这个函数的作用是将输入数据转换成合适的形状。
        np.ndarray -> np.ndarray
        '''
        return x

    @timing
    def fit(self, X_train, y_train):
        self.model, self.optimizer, self.criterion = self.flash_training()
        x = torch.from_numpy(self.xpip(X_train))
        y = torch.from_numpy(y_train)

        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (x, y) in enumerate(dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, torch.argmax(y, dim=1))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == torch.argmax(y, dim=1)).sum().item()

            avg_loss = running_loss / len(dataloader)
            accuracy = 100 * correct / total

            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')

    @timing
    def predict(self, X_test):
        x = torch.from_numpy(self.xpip(X_test))

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    @timing
    def predict_proba(self, X_test):
        x = torch.from_numpy(self.xpip(X_test))

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(x)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs.numpy()
