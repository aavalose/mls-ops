# lab8app/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
# Load your model from MLflow
MODEL_URI = "runs:/13dd6dfeacda43b78e48ee3deb4385c8/model"  
model = mlflow.pyfunc.load_model(MODEL_URI)

# Define the FastAPI app
app = FastAPI()

# Define input data format
class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add all the features your model expects

@app.post("/predict")
def predict(input: InputData):
    # Convert input into a DataFrame
    input_df = pd.DataFrame([input.dict()])
    
    # Predict
    prediction = model.predict(input_df)
    
    return {"prediction": prediction.tolist()}