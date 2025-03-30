from sklearn.ensemble import RandomForestClassifier
from .base import SklearnClfs, astag


@astag
class random_forest(SklearnClfs):
    def __init__(self, **kwargs):
        super(random_forest, self).__init__()

        self.model = RandomForestClassifier(**kwargs)
