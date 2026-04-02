import pytest
import pandas as pd
import numpy as np
from app import rf_classifier, X # This "borrows" the brain from your app

def test_prediction_range():
    """Test if the model gives a valid percentage (between 0 and 100)"""
    # 1. Create a fake patient
    fake_patient = pd.DataFrame([X.iloc[0]]) 
    
    # 2. Ask the model for a prediction
    prediction = rf_classifier.predict_proba(fake_patient)[0][1]
    
    # 3. CHECK: Is the probability between 0 and 1?
    assert 0 <= prediction <= 1
    print(f"Success! Prediction was {prediction}")

def test_extreme_age():
    """Test if the model can handle a very old patient"""
    extreme_patient = pd.DataFrame([X.iloc[0]])
    extreme_patient['age'] = 110 # Set age to 110
    
    prediction = rf_classifier.predict_proba(extreme_patient)[0][1]
    
    assert prediction is not None