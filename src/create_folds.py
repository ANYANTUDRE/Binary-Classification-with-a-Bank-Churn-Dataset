import os

import config

import pandas as pd
from sklearn.model_selection import StratifiedKFold
  #StratifiedGroupKFold


if __name__ == "__main__":
    df = pd.read_csv(config.BASE_TRAINING_FILE)

    # create new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch target
    y = df.Exited.values

    # initiate the Stratified Kfold class from model_selection module
    skf = StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
        df.to_csv("../input/train_folds.csv", index=False)
    