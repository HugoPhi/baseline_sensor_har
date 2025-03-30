import yaml
from plugins.lrkit import ClfTrait, timing


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


class SklearnClfs(ClfTrait):
    def __init__(self):
        super(SklearnClfs, self).__init__()

        self.model = None
        self.training_time = -1
        self.testing_time = -1

    @timing
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())

    @timing
    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred.squeeze()

    @timing
    def predict_proba(self, x_test):
        y_pred = self.model.predict_proba(x_test)
        return y_pred
