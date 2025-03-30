from xgboost import XGBClassifier
from .base import SklearnClfs, astag


@astag
class xgboost(SklearnClfs):
    def __init__(self, **kwargs):
        super(xgboost, self).__init__()

        self.model = XGBClassifier(**kwargs)
