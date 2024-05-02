from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


class HouseInfo(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float


app = FastAPI()

model = joblib.load('ensemble_regression_model.pkl')


@app.post('/predict')
def predict_price(house: HouseInfo):
    data = pd.DataFrame([dict(house)])
    prediction = model.predict(data)
    return {"predicted_price": prediction[0]}

