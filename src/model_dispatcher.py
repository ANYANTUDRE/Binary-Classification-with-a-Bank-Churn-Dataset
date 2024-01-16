from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgbm
import catboost

models = {
    #"logreg": linear_model.LogisticRegression(),
    #"decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    #"decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    #"rf": ensemble.RandomForestClassifier(),  # pas mal mais long a entrainer
    "hist": ensemble.HistGradientBoostingClassifier(),  # bon score, rivalise avec XGBoost
    #"extra": ensemble.ExtraTreesClassifier(), # nope, pas mieux que les autres gbdt
    "xgb": xgb.XGBClassifier(),            # plutot pas mal
    "lgbm": lgbm.LGBMClassifier(n_jobs=-1),
    'gbm': ensemble.GradientBoostingClassifier(),
    'cat': catboost.CatBoostClassifier(),
}