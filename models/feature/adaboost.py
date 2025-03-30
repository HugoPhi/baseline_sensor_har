from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from .base import SklearnClfs, astag


DecisionTreeClassifier = astag(DecisionTreeClassifier)


@astag
class adaboost(SklearnClfs):
    def __init__(self, **kwargs):
        super(adaboost, self).__init__()

        # self.model = AdaBoostClassifier(**kwargs, estimator=DecisionTreeClassifier(max_depth=7))
        self.model = AdaBoostClassifier(**kwargs)
