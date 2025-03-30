import yaml
from plugins.lrkit import ClfTrait, timing

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


def astag(cls):
    tag = f'!{cls.__name__}'

    def constructor(loader, node):
        params = loader.construct_mapping(node)
        return cls(**params)

    yaml.add_constructor(tag, constructor)
    return cls


DecisionTreeClassifier = astag(DecisionTreeClassifier)


class SklearnClfs(ClfTrait):
    def __init__(self):
        super(SklearnClfs, self).__init__()

        self.model = None
        self.training_time = -1
        self.testing_time = -1

    @timing
    def fit(self, X_train, y_train, load=False):
        self.model.fit(X_train, y_train.ravel())

    @timing
    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred.squeeze()

    @timing
    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred


@astag
class DecisionTreeClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(DecisionTreeClf, self).__init__()
        self.model = DecisionTreeClassifier(**kwargs)


@astag
class RandomForestClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(RandomForestClf, self).__init__()

        self.model = RandomForestClassifier(**kwargs)


@astag
class XGBClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(XGBClf, self).__init__()

        self.model = XGBClassifier(**kwargs)


@astag
class AdaBoostClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(AdaBoostClf, self).__init__()

        # self.model = AdaBoostClassifier(**kwargs, estimator=DecisionTreeClassifier(max_depth=7))
        self.model = AdaBoostClassifier(**kwargs)


@astag
class LGBMClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(LGBMClf, self).__init__()

        self.model = LGBMClassifier(**kwargs)


@astag
class SVClf(SklearnClfs):
    def __init__(self, **kwargs):
        super(SVClf, self).__init__()

        self.model = SVC(**kwargs)
