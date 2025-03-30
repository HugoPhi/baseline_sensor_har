from lightgbm import LGBMClassifier
from .base import SklearnClfs, astag


@astag
class lightgbm(SklearnClfs):
    def __init__(self, **kwargs):
        super(lightgbm, self).__init__()

        self.model = LGBMClassifier(**kwargs)
