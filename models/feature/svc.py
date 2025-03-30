from sklearn.svm import SVC
from .base import SklearnClfs, astag


@astag
class svc(SklearnClfs):
    def __init__(self, **kwargs):
        super(svc, self).__init__()

        self.model = SVC(**kwargs)
