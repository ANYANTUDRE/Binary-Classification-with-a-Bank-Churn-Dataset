import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
import config

if __name__ == "__main__":
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    mapping_Geography = {"France": 0, "Spain": 1, "Germany": 2}
    mapping_Gender = {"Male": 0, "Female": 1}
    df.loc[:, "Geography"] = df.Geography.map(mapping_Geography)
    df.loc[:, "Gender"] = df.Gender.map(mapping_Gender)

    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                 "HasCrCard", "IsActiveMember", "EstimatedSalary",
                "Geography", "Gender"]
    X = df[features].values
    y = df.Exited.values

    classifier = ensemble.HistGradientBoostingClassifier(max_iter=500)
    param_grid = {
    "max_leaf_nodes": [None, 20, 50, 30, 40, 60, 100],
    "max_depth":   [1, 2, 5, 7, 11, 13, 15, 20],
    "learning_rate":  np.arange(0.01, 1, 0.01),
    "max_leaf_nodes":  np.arange(20, 150, 10),
    "l2_regularization":  np.arange(0, 10, 1),
    }

    model = model_selection.RandomizedSearchCV( estimator=classifier, 
                                                param_distributions=param_grid, 
                                                n_iter=20,
                                                scoring="roc_auc",
                                                verbose=10, 
                                                n_jobs=1,
                                                cv=5)

    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")