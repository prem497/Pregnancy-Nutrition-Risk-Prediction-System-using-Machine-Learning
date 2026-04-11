from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI(title="Pregnancy Nutrition Risk Prediction API")

# Path setups
BASE_DIR = r"d:\"resume rag bot"
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
ACCURACY_PATH = os.path.join(BASE_DIR, 'model', 'accuracy.pkl')

def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

class HealthData(BaseModel):
    age: int
    bmi: float
    hemoglobin: float
    blood_pressure: int
    sugar_level: int
    protein_intake: int

@app.get("/")
def read_root():
    return {"message": "API Running"}

@app.get("/accuracy")
def get_accuracy():
    acc = load_pkl(ACCURACY_PATH)
    if acc is None:
        raise HTTPException(status_code=404, detail="Accuracy file not found. Please train the model first.")
    return {"accuracy": acc}

@app.post("/predict")
def predict_risk(data: HealthData):
    model = load_pkl(MODEL_PATH)
    acc = load_pkl(ACCURACY_PATH)
    
    if model is None or acc is None:
        raise HTTPException(status_code=404, detail="Model files not found. Please train the model first.")
        
    try:
        # Create input array
        features = np.array([[
            data.age,
            data.bmi,
            data.hemoglobin,
            data.blood_pressure,
            data.sugar_level,
            data.protein_intake
        ]])
        
        prediction = model.predict(features)[0]
        
        return {
            "prediction": prediction,
            "accuracy": acc
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
