import pandas as pd
from typing import Union
from os import getcwd

try:
    from train import *
except ModuleNotFoundError:
    from model.train import *

def load_file(filename : str, sep : str = ",") -> pd.DataFrame:
    """
    Loads a file into a pandas dataframe. Returns an error is file is not a CSV file.

    Parameters:
        filename (str): the file name
        sep (str): the CSV separator

    Returns:
        pd.DataFrame: a pandas dataframe containing the loaded content
    """

    if not filename.endswith(".csv"):
        raise ValueError("File must be a CSV file.")
    
    return pd.read_csv(filename, sep=sep)

def execute_flow(data : dict, target: Union[str, list], model_name : str, replace_nan : str = "none", test_frac : float = 0.2, 
                 test_data : dict = None, resampling_mode : str = "SMOTE", feature_perc : float = 1.0,
                 training_type : str = "auto", 
                 balance : bool = False, select_features : bool = False, trial_num : int = 20) -> list:
    """
    Main function, which automatically executes all steps in a common machine learning model training,
    including data processing, feature selection, training, data balancing, evaluation, and model saving.

    Parameters:
        data (dict): A dictionary containing the training data.
        target (str or list): The target column(s) to separate and encode if necessary.
        model_name (str): Name of the model to train.
        replace_nan (str): Strategy for handling NaN values. Options are:
                           - "none": Drop rows with NaN values.
                           - "zero": Fill NaN values with 0 for numeric columns and "unknown" for non-numeric columns.
                           - "mean": Fill NaN values with column means for numeric columns and "unknown" for non-numeric columns.
        test_frac (float): The amount of the dataset to be used as test set (used when no test data is given).
        test_data (dict): A dictionary containing the test data.
        resampling_mode (str): The sampling method used to balance data (used when 'balance' is set to True).
        feature_perc (float): A value between 0 and 1 indicating the percentage of features to be preserved (used when 'select_features' is set to True).
        training_type (str): Selects the training mode. The available options are:
                        "continuous": indicates the training of a regression model
                        "categorical": indicates the training of a classification model
                        "auto": same as "categorical"
        balance (bool): A flag which indicates the model to balance the data. If set to True, the training data will be balanced according to 'resampling_mode'.
        select_features (bool): A flag which indicates the model to select the best features. If set to True, the model will select the best features according to 'feature_perc'.
        trial_num (int): Number of Optuna trials for hyperparameter tuning.

    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, f1_score).
    """

    try:
        # 1. Processing data

        df = process_data(data, target, replace_nan)
        if test_data:
            test_df = process_data(test_data, target, replace_nan)
            test_data = test_df

        # 2. Split processed data

        X_train, y_train, X_test, y_test = split_data(df, target, test_data, test_frac)

        # 3. Select best features

        if select_features:
            X_train = select_features(X_train, y_train, training_type, feature_perc)
            X_test = X_test[list(X_train.columns)]

        # 4. Balance data

        if balance:
            X_train, y_train = balance_data(X_train, y_train, resampling_mode)
        
        # 5. Training and evaluating selected model (model is saved in this step)

        metrics = train_model(X_train, y_train, X_test, y_test, model_name, mode=training_type, trial_num=trial_num)
        
        return metrics

    except Exception as e:
        print(e)

if __name__ == "__main__":

    dataframe = load_file("iris.data.csv")
    target = list(dataframe.columns)[-1]

    metrics = execute_flow(dataframe.to_dict(orient = "list"), target, model_name="Random Forest Classifier")

    print("Metrics for best trained model:")
    for key, value in metrics.items():
        print(f"{key} : {value}")