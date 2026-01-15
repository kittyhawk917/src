#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
import keras
from keras import layers, callbacks, ops, metrics
import matplotlib.pyplot as plt


def data_preprocessing(df):

    df.drop(['Id', 'LotFrontage', 'GarageYrBlt'], axis=1, inplace=True)

    # set 'MSSubClass' to categorical data
    df['MSSubClass'] = df['MSSubClass'].astype(object)

    # convert 'OverallQual' and 'OverallCond' (rating from 1 to 10) to 0 to 1
    df['OverallQual'] = df['OverallQual'] / 10
    df['OverallCond'] = df['OverallCond'] / 10

    # convert 'NA'/'None' to meaningful labels
    df.replace({'Alley':'NA'}, 'NoAlleyAccess', inplace=True)
    df.replace(
        {'BsmtQual':'NA', 'BsmtCond':'NA', 'BsmtExposure':'NA', 'BsmtFinType1':'NA', 'BsmtFinType2':'NA'}, 
        'NoBasement', inplace=True)
    df.replace({'FireplaceQu':'NA'}, 'NoFireplace', inplace=True)
    df.replace(
        {'GarageType':'NA', 'GarageFinish': 'NA', 'GarageQual': 'NA', 'GarageCond': 'NA'}, 
        'NoGarage', inplace=True)
    df.replace({'PoolQC':'NA'}, 'NoPool', inplace=True)
    df.replace({'Fence':'NA'}, 'NoFence', inplace=True)
    df.replace({'MiscFeature':'NA'}, 'NoMiscFeature', inplace=True)
    df.replace({'MasVnrType':['NA','None']}, 'NoMasonryVeneer', inplace=True)
    
    # replace numeric 'NA' with 0
    feature_numeric_columns = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
    df[feature_numeric_columns] = df[feature_numeric_columns].replace('NA', 0)
    df[feature_numeric_columns] = df[feature_numeric_columns].astype('int64')

    return df


def root_mean_squared_error(y_true, y_pred):

    # Keras 3 (multi-backend)
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))


def neural_network_ames_housing():

    # Load data
    pwd = Path(__file__).parent

    df_train = pd.read_csv(
        Path(pwd, "data/train.csv.gz"), 
        keep_default_na=False,
        na_values=[" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "n/a", "nan", "null "])

    df_test = pd.read_csv(
        Path(pwd, "data/test.csv.gz"),
        keep_default_na=False,
        na_values=[" ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "n/a", "nan", "null "])


    df_train_clean = data_preprocessing(df_train.copy())
    df_test_clean = data_preprocessing(df_test.copy())
    

    # Create training and validation splits
    X_train = df_train_clean.sample(frac=0.7, random_state=42)
    X_valid = df_train_clean.drop(X_train.index)
    y_train = X_train.pop('SalePrice').to_frame()
    y_valid = X_valid.pop('SalePrice').to_frame()

    
    # Preprocess the features
    preprocessor_x = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=object)),
    )
    X_train = preprocessor_x.fit_transform(X_train)
    X_valid = preprocessor_x.transform(X_valid)
    X_test = preprocessor_x.transform(df_test_clean)


    # Scale the target variable
    preprocessor_y = StandardScaler()
    y_train = preprocessor_y.fit_transform(y_train)
    y_valid = preprocessor_y.transform(y_valid)


    # Define the model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(1),
    ])


    model.compile(
        optimizer='adam',
        loss=root_mean_squared_error,
        metrics=[metrics.RootMeanSquaredError]
    )


    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,
        patience=20,
        restore_best_weights=True,
    )


    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=200,
        callbacks=[early_stopping],
        verbose=0,
    )

    history_df = pd.DataFrame(history.history)
    print(f"Minimum validation loss: \n{history_df[['loss', 'val_loss']].min()}")


    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    # Perform inference on the test data
    y_pred = model.predict(X_test)
    y_pred_original = preprocessor_y.inverse_transform(y_pred)

    # Output prediction
    df_output = pd.DataFrame({
        "Id": df_test["Id"], 
        "SalePrice": y_pred_original.tolist()
    })
    df_output["SalePrice"] = df_output["SalePrice"].apply(lambda x: x.pop())


def main() -> int:

    neural_network_ames_housing()

    return 0


if __name__ == '__main__':

    sys.exit(main())

