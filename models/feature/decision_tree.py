from sklearn.tree import DecisionTreeClassifier
from .base import SklearnClfs, astag


@astag
class decision_tree(SklearnClfs):
    def __init__(self, **kwargs):
        super(decision_tree, self).__init__()

        self.model = DecisionTreeClassifier(**kwargs)
