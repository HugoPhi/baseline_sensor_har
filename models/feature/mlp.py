from sklearn.neural_network import MLPClassifier
from .base import SklearnClfs, astag


@astag
class mlp(SklearnClfs):
    def __init__(self, **kwargs):
        super(mlp, self).__init__()

        self.model = MLPClassifier(**kwargs)
