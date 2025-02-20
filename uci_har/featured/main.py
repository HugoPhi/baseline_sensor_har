import toml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from clfs import MLClfs, MLPClf, XGBClfs
from data_process import X_train, y_train, X_test, y_test
from utils import Metrics
from models import MLP


config = toml.load('./hyper.toml')
ml_models = {
    "DecisionTree": DecisionTreeClassifier(**config["DecisionTree"]),
    "RandomForest": RandomForestClassifier(**config["RandomForest"]),
    "XGBoost": XGBClassifier(**config["XGBoost"]),
    "AdaBoost": AdaBoostClassifier(**config["AdaBoost"]),
    "SVM": SVC(**config["SVM"]),
    "LightGBM": LGBMClassifier(**config["LightGBM"]),
}

clf_list = dict()
for name, model in ml_models.items():
    if name == "XGBoost":
        clf_list[name] = XGBClfs(model)
    else:
        clf_list[name] = MLClfs(model)

clf_list["MLP"] = MLPClf(config["mlp"]["epochs"], config["mlp"]["lr"], model=MLP(input_size=X_train.shape[1], output_size=6))


# TODO: 执行器，使用迭代器模式
class Excuter:
    def __init__(self, clf_dict, X_test, metric_list):
        pass


model_res = dict()
for name in clf_list.keys():
    print(f">> {name}: ", end="")
    clf_list[name].fit(X_train, y_train)
    print(f"{clf_list[name].training_time:.4f} s")

    y_pred = clf_list[name].predict_proba(X_test)

    # acc = (y_pred == y_test.reshape(-1)).mean()
    # macro_f1 = f1_score(y_test, y_pred, average="macro")
    # micro_f1 = f1_score(y_test, y_pred, average="micro")
    # model_res[name] = {"acc": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}
    mtc = Metrics(y_test - 1, y_pred)
    print(mtc.f1())
    # rocs = mtc.roc()
    # for (x, y) in rocs:
    #     plt.plot(y, x)
    #     plt.show()
