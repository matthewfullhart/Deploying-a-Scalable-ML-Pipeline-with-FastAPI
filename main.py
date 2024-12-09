import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

path = "model/encoder.pkl"
encoder = load_model(path)

path = "model/model.pkl"
model = load_model(path)

#Creates a RESTful API using FastAPI
app = FastAPI()

#create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    return{"greeting": "Welcome to the API for the Census ML Model!\n-------------------------------------------\n"


# Creates a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # Turn the Pydantic model into a dict, in anticipation of changing it again into a data frame
    data_dict = data.dict()
    # The data has names with hyphens and Python does not allow those as variable names.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features = cat_features,
        encoder = encoder,
        training = false
    )
    _inference = inference(model, data_processed)
    
    return {"result": apply_label(_inference)}
