import joblib
import os
import pandas as pd
import config


def predict(models):
    test  = pd.read_csv(config.TEST_FILE)
    sample = pd.read_csv(config.SAMPLE_FILE)
    predictions = None

    mapping_Geography = {"France": 0, "Spain": 1, "Germany": 2}
    mapping_Gender = {"Male": 0, "Female": 1}
    test.loc[:, "Geography"] = test.Geography.map(mapping_Geography)
    test.loc[:, "Gender"] = test.Gender.map(mapping_Gender)

    features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                 "HasCrCard", "IsActiveMember", "EstimatedSalary",
                "Geography", "Gender"]
    test = test[features].values

    for model in models:
        for fold in range(5):
            clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.pkl"))
            preds = clf.predict_proba(test)[:, 1]
            if fold == 0:
                predictions = preds
            else:
                predictions += preds
        predictions = predictions / 5

        if model == 'hist':
            ens_preds = predictions
        else:
            ens_preds += predictions

    ens_preds = ens_preds / 5

    sample.Exited = ens_preds.astype(float)
    print(sample.head())
    return sample


if __name__ == "__main__":
    models = ["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(models)
    submission.to_csv(f"../output/ensemble_submission.csv", index=False)