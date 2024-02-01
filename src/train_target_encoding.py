import os, argparse, copy, config, itertools, joblib
import model_dispatcher
import pandas as pd 
from sklearn import metrics
from sklearn import preprocessing


def mean_target_encoding(data):
    # make a copy of dataframe
    df = copy.deepcopy(data)
    # list of numerical columns
    num_cols = ['CreditScore', 'Age', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    #cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'Tenure']
    # add new features
    #df = feature_engineering(df, cat_cols)
    features = [f for f in df.columns if f not in ("id", "CustomerId", "Surname", "kfold", "Exited")
                and f not in num_cols]

    # now its time to label encode the features
    for col in features:
        if col not in num_cols: 
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])

    # a list to store 5 validation dataframes
    encoded_dfs = []
    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all feature columns, i.e. categorical columns
        for column in features:
            # create dict of category:mean target
            mapping_dict = dict(
            df_train.groupby(column)["Exited"].mean()
            )
            # column_enc is the new column we have with mean encoding
            df_valid.loc[
            :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
        :, 
        c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df


def run(fold, model):
    # read the training data with folds
    

    #num_cols = ['CreditScore', 'Age', 'Balance', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    #cat_cols = ['Geography', 'Gender', 'NumOfProducts', 'Tenure']
    # add new features
    #df = feature_engineering(df, cat_cols)
    features = [f for f in df.columns if f not in ("id", "CustomerId", "Surname", "kfold", "Exited")]

    # now its time to label encode the features
    """for col in features:
        if col not in num_cols: 
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])"""

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    y_train = df_train.Exited.values

    x_valid = df_valid[features].values
    y_valid = df_valid.Exited.values

    # initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict_proba(x_valid)[:, 1]

    # calculate roc auc
    auc = metrics.roc_auc_score(y_valid, preds)
    print(f"Fold--->{fold}, AUC score={auc}")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.pkl"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    df = pd.read_csv(config.TRAINING_FILE)
    df = mean_target_encoding(df)

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)