import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the serialized model and metadata

def model_load():
    try:
        # Ensure proper loading of the tuple (model, metadata)
        loaded_object = joblib.load('gb_model_metadata.pkl')
        
        # If the loaded object is a tuple, unpack it
        if isinstance(loaded_object, tuple) and len(loaded_object) == 2:
            loaded_model, loaded_metadata = loaded_object
        else:
            raise ValueError("Unexpected format in the loaded model file")

        return loaded_model, loaded_metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Predict using the loaded model
def predict_model(model, data):
    try:
        y_predict = model.predict(data)
        y_predict_proba = model.predict_proba(data)[:, 1]  # For binary classification
        return y_predict, y_predict_proba
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Load the model and metadata
loaded_model, loaded_metadata = model_load()

# Debugging print to check if the model is loaded correctly
print(f"Loaded model type: {type(loaded_model)}")

# Create a FastAPI app
app = FastAPI()

# Define the input model based on your data columns
class InputData(BaseModel):
            satisfaction_level: float
            last_evaluation: float
            number_project: int
            average_montly_hours: int
            time_spend_company: int
            Work_accident: int
            promotion_last_5years: int
            salary: str

@app.post("/predict")
def predict(data: InputData):
    # Encoding the salary feature
    salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
    salary = salary_mapping.get(data.salary, -1)

    if salary == -1:
        raise HTTPException(status_code=400, detail="Invalid salary value")

    # Convert input data into a DataFrame
    df = pd.DataFrame([{
        "satisfaction_level": data.satisfaction_level,
        "last_evaluation": data.last_evaluation,
        "number_project": data.number_project,
        "average_montly_hours": data.average_montly_hours,
        "time_spend_company": data.time_spend_company,
        "Work_accident": data.Work_accident,
        "promotion_last_5years": data.promotion_last_5years,
        "salary": salary
    }])

    # Ensure the model has predict() method
    if not hasattr(loaded_model, "predict"):
        raise HTTPException(status_code=500, detail="Loaded model is not a valid predictor")

    # Predict using the loaded model
    prediction, y_predict_proba = predict_model(loaded_model, df)
    
    return {"prediction": int(prediction[0]), "probability": float(y_predict_proba[0])}


@app.get("/metadata")
def get_metadata():
    """Returns model metadata including features, model type, and other info."""
    
    if not loaded_metadata:
        raise HTTPException(status_code=404, detail="Metadata not available")
    
    metadata_info = {
        "model_type": str(type(loaded_model)), 
        "features": loaded_metadata.get("features", "Not available"),  
        "training_data_shape": loaded_metadata.get("data_shape", "Unknown"), 
        "accuracy": loaded_metadata.get("model_evaluation_accuracy", "Unknown"), 
        "feature_importance": loaded_metadata.get("feature_importance", {}),
        "hyperparameters": loaded_metadata.get("hyperparameters", {}),
        "class_distribution": loaded_metadata.get("class_distribution", {}),
        "model_version": loaded_metadata.get("model_version", "Unknown"),
        "author": loaded_metadata.get("author", "Unknown"),
        "training_timestamp": loaded_metadata.get("training_timestamp", "Unknown"),
    }
    
    return metadata_info


# Run the app with `uvicorn`:
# uvicorn main:app --reload
