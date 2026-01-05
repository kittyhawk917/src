#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_validate

def random_forest():

    # load dataset
    pwd = Path(__file__).parent
    df_train = pd.read_csv(Path(pwd, "data/train.csv"))
    df_test = pd.read_csv(Path(pwd, "data/test.csv"))


    # create a new column indicating missing values
    df_train["Age_Missed"] = df_train["Age"].isna()
    df_test["Age_Missed"] = df_test["Age"].isna()

    # impute missing values with mean (train)
    imputer_mean = SimpleImputer(strategy='mean')
    df_train_age_imputed = imputer_mean.fit_transform(df_train[["Age"]])
    df_train[["Age"]] = df_train_age_imputed
    
    # age mean in training data
    df_train_age_mean = imputer_mean.statistics_[0]

    # impute missing values with mean (test)
    df_test.fillna(value={"Age": df_train_age_mean}, inplace=True)


    # features
    feature_col_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Age_Missed"]
    target_col_name = "Survived"

    X_train = pd.get_dummies(df_train[feature_col_names])

    y_train = df_train[target_col_name]

    X_test = pd.get_dummies(df_test[feature_col_names])


    # model
    rf = RandomForestClassifier(random_state=42)
    

    # cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    scoring_metrics = ['accuracy', 'precision', 'f1', 'recall', 'roc_auc']


    # grid search
    param_grid = [
        {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [2, 3],
            "bootstrap": [True, False],
        }
    ]
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=skf, scoring=scoring_metrics, refit='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")
    print(f"Best Estimator: {grid_search.best_estimator_}")


    # update the model
    rf = grid_search.best_estimator_
    rf.fit(X_train, y_train)

    
    # predict class for test data
    y_pred = rf.predict(X_test)
    
    # output prediction
    df_output = pd.DataFrame({"PassengerId": df_test["PassengerId"], 'Survived': y_pred})


def main() -> int:

    random_forest()

    return 0


if __name__ == '__main__':
    sys.exit(main())

