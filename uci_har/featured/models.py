import torch as tc


class MLP(tc.nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = tc.nn.Linear(input_size, 128)
        self.fc2 = tc.nn.Linear(128, 64)
        self.fc3 = tc.nn.Linear(64, 32)
        self.fc4 = tc.nn.Linear(32, output_size)

    def forward(self, x):
        x = tc.relu(self.fc1(x))
        x = tc.relu(self.fc2(x))
        x = tc.relu(self.fc3(x))
        x = self.fc4(x)  # CrossEntropyLoss会自动处理logits
        return x
