from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from typing import Union
from types import NoneType
import zipfile
from model.pipeline import execute_flow

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class DownloadRequest(BaseModel):
    files: list

class TrainingRequest(BaseModel):
    data: dict
    target: Union[str, list]
    model_name: str
    replace_nan: str = "none"
    test_frac: float = 0.2
    test_data: Union[dict, NoneType] = None
    resampling_mode: str = "SMOTE"
    feature_perc: float = 1.0
    training_type: str = "auto"
    balance: bool = False
    select_features: bool = False
    trial_num: int = 20


# Define routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the Model Testing App!"}

@app.post("/download/")
def download(request: DownloadRequest):

    # Compress files into a zip file
    with zipfile.ZipFile("temp/model_schema.zip", "w") as zipf:
        for file in list(request.files):
            zipf.write(file)
    os.remove("temp/schema.txt")
    return {"status": "Success", "file_name" : "model_schema.zip", "type": "ZIP"}

@app.post("/train/")
def train(request: TrainingRequest):

    # Train model
    try:
        metrics = execute_flow(request.data, request.target, request.model_name, 
                               replace_nan=request.replace_nan, test_frac=request.test_frac, 
                               test_data=request.test_data, resampling_mode=request.resampling_mode,
                               feature_perc=request.feature_perc, training_type=request.training_type, 
                               balance=request.balance, select_features=request.select_features, 
                               trial_num=request.trial_num)
        
        return  {"status" : "Success",
                    "trained_model" : request.model_name,
                    "metrics": metrics}
    
    except Exception as e:
        return {"status": "Failed",
                "error_content": str(e)}