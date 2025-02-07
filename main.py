import streamlit as st
import pandas as pd
import json
from model.pipeline import *
import requests
from pathlib import Path
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
import base64
import time
import os
import shutil

# macros

DEFAULT_TRIALS = 20
DEFAULT_RESAMPLING = "SMOTE"
DEFAULT_TRAINING_MODE = "auto"
DEFAULT_FEATURE_FRAC = 1.0


def get_download_link(file_path : str, file_label: str="Download file"):
    """
    Auxiliary function to create a download button (simpler working approach to deal with streamlit's download button)
    """
    with open(file_path, "rb") as f:
        file_data = f.read()
        b64 = base64.b64encode(file_data).decode()  # Encode file to base64
        href = f"""<a href="data:application/octet-stream;base64,{b64}" download="{file_path.split("/")[-1]}"
            style="text-decoration: none;"><button style="background-color: #FF4B4B;
                    border: none;
                    color: white;
                    padding: 15px 32px;
                    text-align: center;
                    display: inline-block;
                    font-size: 16px;
                    border-radius: 8px;
                    cursor: pointer;">
            {file_label}</button></a>"""
        return href

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def reset_page():
    st.session_state.rerun_flag = True
    # removes temporary files when page is reset
    if os.path.exists("temp/model_schema.zip"):
        os.remove("temp/model_schema.zip")
    if os.path.exists("api/model.pkl"):
        os.remove("api/model.pkl")

def send_post_request(payload, url):
    """Send POST request and return response."""
    json_payload = json.dumps(payload)
    return requests.post(url, data=json_payload, headers={"Content-Type": "application/json"})

def write_sidebar(**kwargs):

    resampling_mode = kwargs["resampling_mode"]
    feature_perc = kwargs["feature_perc"]
    training_type = kwargs["training_type"]
    response, payload = None, None

    st.write("Loaded dataframe head:")
    st.write(kwargs["data"].head())

    if kwargs["test_data"] is not None:
        st.write("Loaded test data head:")
        st.write(kwargs["test_data"].head())

    # Starting configuration
    st.write("Training configuration")

    # 1. Target variable
    target = st.multiselect("Select target column(s)...", list(kwargs["data"].columns))

    # 2. Handling missing data
    options = ["none", "zero", "mean"]
    replace_nan = st.selectbox("Replace missing data with:", options)

    # 3. Train-test split
    if kwargs["test_data"] is None:
        test_frac = st.slider("Select amount of data to be used to test (1 is entire data)", 0., 1., 0.2)

    # 4. Dealing with class imbalance
    balance = st.checkbox("I want to balance the classes")
    if balance:
        resampling_options = ["Random Over-Sampler", "SMOTE", "Random Under-Sampler", "NearMiss"]
        resampling_mode = st.selectbox("Data resampling approach:", resampling_options)

    # 5. Feature selection
    select_features = st.checkbox("I want to perform feature selection")
    if select_features:
        feature_perc = st.slider("Select amount of features to be preserved (1 is all features)", 0., 1., 1.)

    # 6. Model selection
    available_models = ["Random Forest Classifier", "Ada Boost Classifier", "Random Forest Regressor"]
    model_name = st.selectbox("Select a model:", available_models)

    if model_name in ["Random Forest Regressor"]:
        training_type = "continuous"

    # 7. Set number of trials
    trial_num = st.slider("Select number of trials:", 1, 100, 20)

    # 8. Training button
    train_button = st.button("Train", disabled=not (kwargs["data"] is not None and model_name and len(target) > 0))

    # if training button is displayed
    if train_button:

        payload = {
            "data": kwargs["data"].to_dict(orient="list"),
            "target": target,
            "model_name": model_name,
            "replace_nan": replace_nan,
            "test_frac": test_frac,
            "test_data": kwargs["test_data"].to_dict(orient="list") if kwargs["test_data"] is not None else kwargs["test_data"],
            "resampling_mode": resampling_mode,
            "feature_perc": feature_perc,
            "training_type": training_type,
            "balance": balance,
            "select_features": select_features,
            "trial_num": trial_num
        }

        with st.spinner("Please be patient. The model is training..."):

            # Make the POST request
            response = send_post_request(payload, kwargs["url"] + "train/"),
    # end of if train_button

    return response, payload

def clear_logs(expr_id: str, preserve: str = "today") -> None:
    """
    Removes logs from an experiment based on a preservation period.

    Parameters:
        expr_id (str): The ID of the experiment.
        preserve (str): The time frame for preserving logs:
                        "latest_hour" - preserves logs from the last hour
                        "today" - preserves logs from the last 24 hours
                        "latest_week" - preserves logs from the last 7 days
                        "none" - deletes all logs

    Returns:
        None
    """
    run_folder = Path("mlruns") / expr_id

    # Check if folder exists
    if not run_folder.exists():
        print(f"Experiment folder {run_folder} does not exist.")
        return

    # Define time threshold for preservation
    current_time = datetime.now()
    if preserve == "latest_hour":
        threshold = current_time - timedelta(hours=1)
    elif preserve == "today":
        threshold = current_time - timedelta(days=1)
    elif preserve == "latest_week":
        threshold = current_time - timedelta(days=7)
    else:  # "none" means delete everything
        threshold = datetime.fromtimestamp(1)  # Very old timestamp

    # Identify runs to delete
    run_list = [(item.name, item.stat().st_ctime) for item in run_folder.iterdir() if item.is_dir()]
    runs_to_delete = [item[0] for item in run_list if datetime.fromtimestamp(item[1]) <= threshold]

    if not runs_to_delete:
        print(f"No logs to delete for experiment {expr_id} (preserve mode: {preserve})")
        return

    print(f"Deleting {len(runs_to_delete)} logs from experiment {expr_id} (preserve mode: {preserve})")

    # Delete identified logs
    for run in runs_to_delete:
        try:
            shutil.rmtree(run_folder / run)
            print(f"Deleted: {run}")
        except Exception as e:
            print(f"Failed to delete {run}: {e}")


def retrieve_logs(expr_id: str, filter_logs: bool = False, log_num: int = 20) -> str:
    """
    Retrieves MLflow logs for a given experiment.

    Parameters:
        expr_id (str): The ID of the experiment.
        filter_logs (bool): If True, only logs from today will be retrieved.
        log_num (int): The number of logs to return.

    Returns:
        str: A formatted string containing experiment logs.
    """
    log_text = ""
    client = MlflowClient()

    # Get current date
    current_date = datetime.today().date()

    try:
        # Retrieve runs for the experiment (sorted by start_time descending)
        runs = client.search_runs([expr_id], order_by=["start_time desc"])

        # Filter by date if enabled
        if filter_logs:
            runs = [run for run in runs if datetime.fromtimestamp(run.info.start_time / 1000).date() == current_date]

        # Limit results
        runs = runs[:log_num]

        # Construct log text
        for run in runs:
            run_id = run.info.run_id
            log_text += f"\n==== Run ID: {run_id} ====\n"
            log_text += f"Name: {run.info.run_name}\n"
            log_text += f"Start date: {time.ctime(run.info.start_time / 1000)}\n"
            log_text += f"End date: {time.ctime(run.info.end_time / 1000)}\n"
            log_text += f"Status: {run.info.status}\n"
            log_text += f"Metrics: {json.dumps(run.data.metrics, indent=4)}\n"
            log_text += f"Parameters: {json.dumps(run.data.params, indent=4)}\n"
            log_text += "\n\n"

    except Exception as e:
        return f"Error retrieving logs: {str(e)}"

    return log_text
                
def generate_schema(flow_args: dict) -> None:
    """
    Generates a text file containing the steps performed by the app to train the model.

    Parameters:
        flow_args (dict): a dictionary containing the model configuration

    Returns:
        None
    """

    text = f"""This is a file containing the performed steps to train the model
    
        Model name: {flow_args["model_name"]}
        Labels: {flow_args["target"]}

        Performed steps:
            -> Data processing
                - Drop duplicates"""

    if flow_args["replace_nan"] == "none":
        text += """
                - Drop rows with null values"""
    elif flow_args["replace_nan"] == "zero":
        text += """
                - Replace null values with zero (numerical columns) or 'unknown' (non-numerical columns)"""
    else:
        text += """
                - Replace null values with mean value of column (numerical columns) or 'unknown' (non-numerical columns)"""
    
    text += """
                - Split feature and target columns
                - One-hot encoding using pd.get_dummies
                - Label encoding using LabelEncoder"""
    
    if flow_args["test_data"]:
        text += """
            -> Test data processing"""
    
    text += """
            -> Split data into training and test parts"""
    
    if flow_args["select_features"]:
        text += """
            -> Performing feature selection using SelectKBest"""

    if flow_args["balance"]:
        text += f"""
            -> Treating class imbalance using {flow_args["resampling_mode"]}"""
        
    text += """
            -> Training and evaluating model. Hyperparameters were tuned using Optuna through trials"""
    
    with open("temp/schema.txt", "w") as f:
        f.write(text)
        
    

def main(preserve = False):

    # general variables
    data = None
    test_data = None
    uploaded_test_file = None
    resampling_mode = DEFAULT_RESAMPLING
    feature_perc = DEFAULT_FEATURE_FRAC
    training_type = DEFAULT_TRAINING_MODE
    trial_num = DEFAULT_TRIALS

    # training related variables
    payload = None
    url = st.secrets["database"]["url"]
    response = None

    # interface related variables
    uploaded_test_file = None
    uploaded_file = None

    if "rerun_flag" not in st.session_state:
        st.session_state.rerun_flag = False

    if st.session_state.rerun_flag:
        st.session_state.rerun_flag = False  # Reset the flag
        st.rerun()

    # App automatically clears logs every time it's initialised
    if not preserve:
        clear_logs("0")

    st.title("SAMLEMT - Simple Automated Machine Learning Model Training")

    st.write("Welcome to SAMLEMT, a simple tool to create machine learning models automatically. All \
             you need to do is loading data and setting training configuration. SAMLEMT takes care of the rest.")

    intro_text = st.empty()
    file_uploader = st.empty()
    enable_test = st.empty()

    intro_text.write("Before we begin, upload your data:")
    uploaded_file = file_uploader.file_uploader("Choose a CSV file (training data)", type=['csv'])
    has_test_data = enable_test.checkbox("I have a separate file to test the model")

    # enable another file uploader if separate test data option is selected
    if has_test_data:
        uploaded_test_file = st.file_uploader("Choose a CSV file (test data)", type=['csv'])

    if uploaded_file:
        
        data = load_csv(uploaded_file)
        test_data = load_csv(uploaded_test_file) if uploaded_test_file else None


        # enables side bar after uploading a CSV file
        with st.sidebar:

            response, payload = write_sidebar(data = data, test_data = test_data, 
                          url = url, resampling_mode = resampling_mode,
                          feature_perc = feature_perc, training_type = training_type)
            
            if response:
                response = response[0]

        # end of with st.sidebar

    # If training pipeline was called
    if response:

        intro_text.empty()
        file_uploader.empty()
        enable_test.empty()

        with st.expander("Training logs"):
            st.code(retrieve_logs("0", filter_logs=True, log_num=trial_num), language=None)

        if response.status_code >= 400:
            colour = "red"
            st.error("An error with the model training has occurred")
        elif response.status_code >= 300:
            colour = "orange"
            st.warning("Attention: one or more warnings were triggered")
        elif response.status_code >= 200:
            colour = "green"
            st.success("The model was successfully trained!")
        else:
            colour = "blue"
        st.markdown(f"Returned training status: :{colour}[{response.status_code}]")
        st.write("Returned message:")
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            response_json = {"error": "Invalid JSON response"}
        st.code(response_json, language="json")

        generate_schema(payload)
        try:
            zipfy_request = send_post_request(payload = {"files": ["api/model.pkl", "temp/schema.txt"]}, url = url + "download/")
            if zipfy_request.status_code != 200:
                raise Exception(f"Could not generate file. Returned error: {zipfy_request.status_code}")
        except Exception as e:
            st.error(str(e))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(get_download_link("temp/model_schema.zip", "Download model"), unsafe_allow_html=True)
        with col2:
            st.button("Train with another data", on_click=reset_page)
    

    st.caption("This project was done by Christopher Brandão.")
    st.caption("For more project like this, visit their [GitHub page](https://www.github.com/CoraPhoenix/)")


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print(f" An error has occurred in 'main': {e.with_traceback(e.__traceback__)}")