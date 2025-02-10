from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

X_train = pd.read_csv(
    "./data/train/X_train.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_train = pd.read_csv(
    "./data/train/y_train.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

X_test = pd.read_csv(
    "./data/test/X_test.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

y_test = pd.read_csv(
    "./data/test/y_test.txt",
    sep=r'\s+',
    header=None,
    engine='python').to_numpy()

shuffle = np.random.permutation(len(X_train))
print(len(X_train), len(X_test))
X_train, y_train = X_train[shuffle], y_train[shuffle]

models = {
    "DecisionTree": DecisionTreeClassifier(),

    "RandomForest": RandomForestClassifier(n_estimators=100),

    "LGBM": LGBMClassifier(n_estimators=100),

    "XGBoost": XGBClassifier(n_estimators=100),

    "AdaBoost": AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=7),
        n_estimators=100
    ),

    "SVM": SVC(),  # 注意：SVM的predict函数返回的是类别标签，需要+1转化为0-indexed

    "LightGBM": LGBMClassifier(n_estimators=100)  # 注意：LightGBM返回的是类别概率，需要取argmax转化为类别标签
}


def train(model="DecisionTree", models=models):
    if model in models.keys():
        if model == "XGBoost":
            models[model].fit(X_train, y_train.ravel() - 1)
            y_pred = models[model].predict(X_test)

            y_pred = y_pred + 1
        else:
            models[model].fit(X_train, y_train.ravel())
            y_pred = models[model].predict(X_test)

    return y_pred


y_pred, train_accuracy = train("DecisionTree")
print((y_pred == y_test.reshape(-1)).mean())
print(classification_report(y_test, y_pred))

# # 决策树
# if False:
#     tree_clf = DecisionTreeClassifier()
#     tree_clf.fit(X_train, y_train.ravel())
#     y_pred = tree_clf.predict(X_test)
#
#
# # 支持向量机
# if False:
#     svm_clf = SVC()
#     svm_clf.fit(X_train, y_train.ravel())
#     y_pred = svm_clf.predict(X_test)
#
#
# # 随机森林
# if False:
#     rf_clf = RandomForestClassifier(n_estimators=100)
#     rf_clf.fit(X_train, y_train.ravel())
#     y_pred = rf_clf.predict(X_test)
#
# # XGBoost
# if False:
#     xgboost_clf = XGBClassifier(n_estimators=100)
#     xgboost_clf.fit(X_train, y_train.ravel() - 1)
#     y_pred = xgboost_clf.predict(X_test)
#
#     y_pred = y_pred + 1
#
# # AdaBoost
# if False:
#     adaboost_clf = AdaBoostClassifier(
#         base_estimator=DecisionTreeClassifier(max_depth=7),
#         n_estimators=100)
#     adaboost_clf.fit(X_train, y_train.ravel())
#     y_pred = adaboost_clf.predict(X_test)
#
#
# # LightGBM
# if False:
#     lgbm_clf = LGBMClassifier(n_estimators=100)
#     lgbm_clf.fit(X_train, y_train.ravel())
#     y_pred = lgbm_clf.predict(X_test)
