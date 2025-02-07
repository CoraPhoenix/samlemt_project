import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_iris
import numpy as np
import types
from typing import Union
import optuna
import mlflow
import joblib

import warnings
warnings.filterwarnings("ignore")

def process_data(dataframe: dict, target: Union[str, list], replace_nan: str = "none") -> pd.DataFrame:
    """
    Processes a dataframe to prepare it for model training.
    
    Parameters:
        dataframe (dict): The input dictionary to process.
        target (str or list): The target column(s) to separate and encode if necessary.
        replace_nan (str): Strategy for handling NaN values. Options are:
                           - "none": Drop rows with NaN values.
                           - "zero": Fill NaN values with 0 for numeric columns and "unknown" for non-numeric columns.
                           - "mean": Fill NaN values with column means for numeric columns and "unknown" for non-numeric columns.
                           
    Returns:
        pd.DataFrame: The processed dataframe with features and target columns.
    """
    # Validate inputs
    if not isinstance(target, (str, list)):
        raise ValueError("Input 'target' must be a string or a list of strings.")
    if replace_nan not in ["none", "zero", "mean"]:
        raise ValueError("Invalid value for 'replace_nan'. Choose from 'none', 'zero', or 'mean'.")
    
    # Ensure target is a list for consistency
    target = [target] if isinstance(target, str) else target

    # Convert dictionary to a pandas dataframe
    dataframe = pd.DataFrame(dataframe)

    # Check if target columns exist
    missing_targets = [col for col in target if col not in dataframe.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in dataframe: {missing_targets}")
    
    # Drop duplicates
    dataframe = dataframe.drop_duplicates()

    # Handle missing values
    if replace_nan == "none":
        dataframe = dataframe.dropna()
    else:
        numerical_cols = dataframe.select_dtypes(include=['number'])
        non_numerical_cols = dataframe.select_dtypes(exclude=['number'])
        
        if replace_nan == "zero":
            dataframe[numerical_cols.columns] = numerical_cols.fillna(0)
            dataframe[non_numerical_cols.columns] = non_numerical_cols.fillna("unknown")
        elif replace_nan == "mean":
            dataframe[numerical_cols.columns] = numerical_cols.fillna(numerical_cols.mean())
            dataframe[non_numerical_cols.columns] = non_numerical_cols.fillna("unknown")

    # Split features and target
    target_df = dataframe[target]
    features_df = dataframe.drop(target, axis=1)

    # One-hot encode categorical features
    features_df = pd.get_dummies(features_df, drop_first=True)

    # Encode target columns if necessary
    target_df = target_df.apply(
        lambda col: LabelEncoder().fit_transform(col) if col.dtypes == "object" else col
    )

    # Combine features and target
    return pd.concat([features_df, target_df], axis=1)


def split_data(train_df: pd.DataFrame, target: Union[str, list], test_df: pd.DataFrame = None, test_frac: float = 0.2) -> list:
    """
    Gets one or two dataframes and split it into training and test parts accordingly. If it gets one
    dataframe only, it'll be split into training and test parts. Otherwise, each dataframe will be
    split into feature (X) and target (y) parts.

    Parameters:
        train_df (pd.DataFrame): The training dataframe to process.
        target (str or list): The target column(s) to separate and encode if necessary.
        test_df (pd.DataFrame): The test dataframe to process.
        test_frac (float): The amount of the dataset to be used as test set.
                           
    Returns:
        list: A list containing the train-test split of inputs
    """

    # Validate inputs
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError("Input 'train_df' must be a pandas DataFrame.")
    if not isinstance(target, (str, list)):
        raise ValueError("Input 'target' must be a string or a list of strings.")
    if not isinstance(test_df, (types.NoneType, pd.DataFrame)):
        raise ValueError("Input 'test' must be a pandas DataFrame or 'None'.")

    X_train, y_train, X_test, y_test = 4*[None]

    # Ensure target is a list for consistency
    target = [target] if isinstance(target, str) else target

    if test_df:
        X_train, y_train = train_df.drop(target, axis=1), train_df[target]
        X_test, y_test = test_df.drop(target, axis=1), test_df[target]
    else:
        X_train, X_test, y_train, y_test = train_test_split(train_df.drop(target, axis=1), train_df[target],
        random_state=42, test_size=test_frac, shuffle=True)

    return X_train, y_train, X_test, y_test


def select_features(X, y, mode : str = "auto", feature_perc: float = 1.0):
    """
    Performs a feature selection, returning training data with the features which represent a given fraction
    of the input data.

    Parameters:
        X (pd.DataFrame or np.array): A dataframe or array containing the training variables.
        y (pd.DataFrame or np.array): A dataframe or array containing the training targets.
        mode (str): Selects the training mode. The available options are:
                        "continuous": indicates the training of a regression model
                        "categorical": indicates the training of a classification model
                        "auto": same as "categorical"
        feature_perc (float): A value between 0 and 1 indicating the percentage of features to be preserved.
                           
    Returns:
        A list, dataframe, or array containing the data with selected features
    """

    if not isinstance(X, (np.array, pd.DataFrame)):
        raise ValueError("Input 'X' must be a pandas DataFrame or a numpy array.")
    
    if feature_perc > 1 or feature_perc < 0:
        raise ValueError("Input 'feature_perc' must be a value between 0 and 1.")
    
    # selector variable
    selector = None

    if mode in ["auto", "categorical"]:
        selector = SelectKBest(k=int(X.shape[1] * feature_perc))
    elif mode == "continuous":
        selector = SelectKBest(score_func=f_regression, k=int(X.shape[1] * feature_perc))
    else:
        raise ValueError(f"Unsupported value for 'mode': {mode}")
    
    return selector.fit_transform(X, y)
    

def balance_data(X, y, sampling_mode: str = "SMOTE"):
    """
    Balances the training data so that the number of samples for every training label is the same.

    Parameters:
        X (pd.DataFrame or np.array): A dataframe or array containing the training variables.
        y (pd.DataFrame or np.array): A dataframe or array containing the training targets.
        sampling_mode (str): The sampling method used to balance data.
                           
    Returns:
        A list, dataframe, or array containing the resampled training data
    """

    if not isinstance(sampling_mode, (str)):
        raise ValueError("Input 'sampling_mode' must be a string.")

    sampling_model = None

    if sampling_mode == "Random Over-Sampler":
        sampling_model = RandomOverSampler()
    elif sampling_mode == "SMOTE":
        sampling_model = SMOTE()
    elif sampling_mode == "Random Under-Sampler":
        sampling_model = RandomUnderSampler()
    elif sampling_mode == "NearMiss":
        sampling_model = NearMiss()

    return sampling_model.fit_resample(X, y)


def train_model(X_train, y_train, X_test, y_test, model_name: str, mode : str = "auto", trial_num:int=20) -> dict:
    """
    Train a model using Optuna for hyperparameter tuning and log results with MLflow.

    Parameters:
        X_train, y_train: Training features and labels.
        X_test, y_test: Testing features and labels.
        model_name (str): Name of the model to train. Options: 
                          "Random Forest Classifier", "Ada Boost Classifier", "Random Forest Regressor".
        mode (str): Selects the training mode. The available options are:
                        "continuous": indicates the training of a regression model
                        "categorical": indicates the training of a classification model
                        "auto": same as "categorical"
        trial_num (int): Number of Optuna trials for hyperparameter tuning.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train = y_train.to_numpy().ravel()
            y_test = y_test.to_numpy().ravel()
    def objective(trial):
        # Initialize model based on model_name
        model = None
        params = {}
        
        if model_name == "Random Forest Classifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            model = RandomForestClassifier(**params, random_state=42)
        
        elif model_name == "Ada Boost Classifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0)
            }
            model = AdaBoostClassifier(**params, random_state=42)
        
        elif model_name == "Random Forest Regressor":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
            model = RandomForestRegressor(**params, random_state=42)
        
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation metric
        if mode in ["auto", "categorical"]:
            individual_accuracies = [accuracy_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])] if len(y_test.shape) > 1 and y_test.shape[1] > 1 \
                else [accuracy_score(y_test, y_pred)]
            average_accuracy = sum(individual_accuracies) / len(individual_accuracies)
            accuracy = average_accuracy

            # Log parameters and metrics to MLflow
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, "model", input_example = X_train.iloc[:1])

            return accuracy

        elif mode == "continuous":
            individual_scores = [r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])] if len(y_test.shape) > 1 and y_test.shape[1] > 1 \
                else [r2_score(y_test, y_pred)]
            average_r2 = sum(individual_scores) / len(individual_scores)
            r2 = average_r2

            # Log parameters and metrics to MLflow
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric("r2 score", r2)
                mlflow.sklearn.log_model(model, "model", input_example = X_train.iloc[:1])

            return r2
        else:
            raise ValueError(f"Unsupported value for 'mode': {mode}")

    # Set up Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trial_num)

    # Get the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Accuracy: {best_trial.value}" if mode != "continuous" else f"  R-squared: {best_trial.value}")
    print(f"  Params: {best_trial.params}")

    # Initialize the best model with optimal parameters
    if model_name == "Random Forest Classifier":
        best_model = RandomForestClassifier(**best_trial.params, random_state=42)
    elif model_name == "Ada Boost Classifier":
        best_model = AdaBoostClassifier(**best_trial.params, random_state=42)
    elif model_name == "Random Forest Regressor":
        best_model = RandomForestRegressor(**best_trial.params, random_state=42)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Train the best model on the entire training set
    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, "api/model.pkl")

    # Evaluate the best model
    y_pred = best_model.predict(X_test)

    if mode in ["auto", "categorical"]:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred, normalize=True),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
    else:
        metrics = {
            "RMSE": mean_squared_error(y_test, y_pred, normalize=True, squared=False),
            "MAE": mean_absolute_error(y_test, y_pred, average="weighted"),
            "r2_score": r2_score(y_test, y_pred, average="weighted")
        }

    return metrics




if __name__ == "__main__":

    #data = load_iris()
    #X = pd.DataFrame(data.data, columns=data.feature_names)
    #y = data.target
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_df = pd.read_csv("iris.data.csv", sep=",")

    print("Base dataframe:\n", test_df.head())

    #test_df = test_df.drop(columns=["G2", "G3"], axis = 1)
    test_df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    test_df = process_data(test_df, ["species"])

    print("Processed dataframe:\n", test_df.head())

    X_train, y_train, X_test, y_test = split_data(test_df, ["species"])

    print(f"Total dataframe shape: {test_df.shape}\nTraining set shapes: {X_train.shape}, {y_train.shape}\nTest set shapes: {X_test.shape}, {y_test.shape}")

    metrics = train_model(X_train, y_train, X_test, y_test, "Random Forest Classifier")

    print("Returned metrics for the best model:")
    for key, value in metrics.items():
        print(f"{key}: {value}")