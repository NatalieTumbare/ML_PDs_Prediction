# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:12:36 2024

@author: ntumbare
"""

#1. Library imports
#Anaconda IDE
#cd C:\Users\ntumbare\Desktop\ML_PDs_Prediction\ML_PDs_Prediction
#pip install fastapi uvicorn
#uvicorn app:app --reload


#http:127.0.0.1:800
#http:127.0.0.1:800/docs


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
#import pandas as pd
#import requests

#2. Create the app object
app = FastAPI()

# Load Logistic Regression model
  
pickle_in = open("final_LR.pkl", "rb")
final_LR = pickle.load(pickle_in)

#Load Random Forest model
pickle_in = open("final_RFC.pkl", "rb")
final_RFC = pickle.load(pickle_in)

# Load Decision Tree model

pickle_in = open("final_DTC.pkl", "rb")
final_DTC = pickle.load(pickle_in)
    
## Load Gradient Boosting model
 
pickle_in = open("final_GBC.pkl", "rb")
final_GBC = pickle.load(pickle_in)

## Load Linear Discriminant model

pickle_in = open("final_LDA.pkl", "rb")
final_LDA = pickle.load(pickle_in)

#3. Index route, opens automatically on http:127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello user'}    

#4. Route with a single parameter, returns the parameter within a 
#Located at: http:127.0.0.1:800/AnyName_ID_Here
@app.get('/{Loan_ID}')
def get_name(Loan_ID: str):
    return {'Welcome to PD prediction': f'{Loan_ID}'}


#5. Define the Pydantic Model for Input Data
class PredictInput(BaseModel):
    loan_amount: float  # Units: Zimbabwe Dollars
    number_of_defaults: float  # Units: Number of defaults
    outstanding_balance: float  # Units: Zimbabwe Dollars
    interest_rate: float  # Units: Percentage (0-100)
    age: float  # Units: Years
    remaining_term: float  # Units: Months
    salary: float  # Units: Zimbabwe Dollars
    gender_female: int  # Units: 0 or 1 (binary)
    gender_male: int  # Units: 0 or 1 (binary)
    gender_other: int  # Units: 0 or 1 (binary)
    marital_status_divorced: int  # Units: 0 or 1 (binary)
    marital_status_married: int  # Units: 0 or 1 (binary)
    marital_status_single: int  # Units: 0 or 1 (binary)
    marital_status_unknown: int  # Units: 0 or 1 (binary)
    province_Bulawayo: int  # Units: 0 or 1 (binary)
    province_Harare: int  # Units: 0 or 1 (binary)
    province_Manicaland: int  # Units: 0 or 1 (binary)
    province_Mashonaland_East: int  # Units: 0 or 1 (binary)
    province_Mashonaland_West: int  # Units: 0 or 1 (binary)
    province_Masvingo: int  # Units: 0 or 1 (binary)
    province_Matabeleland_North: int  # Units: 0 or 1 (binary)
    province_Matabeleland_South: int  # Units: 0 or 1 (binary)
    province_Midlands: int  # Units: 0 or 1 (binary)
    province_Not_Specified: int  # Units: 0 or 1 (binary)	
    
# 4. Create the FastAPI Endpoint
@app.post("/predict")
def predict(input_data: PredictInput):
    input_data = input_data.dict()
    print(input_data)
    print("Hello")
    loan_amount = input_data['loan_amount']
    number_of_defaults = input_data['number_of_defaults']
    outstanding_balance = input_data['outstanding_balance']
    interest_rate = input_data['interest_rate']
    age = input_data['age']
    remaining_term = input_data['remaining_term']
    salary = input_data['salary']
    gender_female = input_data['gender_female']
    gender_male = input_data['gender_male']
    gender_other = input_data['gender_other']
    marital_status_divorced = input_data['marital_status_divorced']
    marital_status_married = input_data['marital_status_married']
    marital_status_single = input_data['marital_status_single']
    marital_status_unknown = input_data['marital_status_unknown']
    province_Bulawayo = input_data['province_Bulawayo']
    province_Harare = input_data['province_Harare']
    province_Manicaland = input_data['province_Manicaland']
    province_Mashonaland_East = input_data['province_Mashonaland_East']
    province_Mashonaland_West = input_data['province_Mashonaland_West']
    province_Masvingo = input_data['province_Masvingo']
    province_Matabeleland_North = input_data['province_Matabeleland_North']
    province_Matabeleland_South = input_data['province_Matabeleland_South']
    province_Midlands = input_data['province_Midlands']
    province_Not_Specified = input_data['province_Not_Specified']
	
    # Convert the input data to a numpy array
    X_input = np.array([[loan_amount, number_of_defaults, outstanding_balance, interest_rate, age, remaining_term, salary, gender_female, gender_male, gender_other, marital_status_divorced, marital_status_married, marital_status_single, marital_status_unknown, province_Bulawayo, province_Harare, province_Manicaland, province_Mashonaland_East, province_Mashonaland_West,  province_Masvingo, province_Matabeleland_North, province_Matabeleland_South, province_Midlands, province_Not_Specified]])

    # Make predictions using the individual models
#    y_pred_LR = final_LR.predict(X_input)[0]
#    y_pred_RFC = final_RFC.predict(X_input)[0]
#    y_pred_DTC = final_DTC.predict(X_input)[0]
#    y_pred_GBC = final_GBC.predict(X_input)[0]
#    y_pred_LDA = final_LDA.predict(X_input)[0]
    
     
 # Convert the predictions to human-readable format
 #   if  y_pred_LR == 1:
 #       y_pred_LR_str = "Defaulted"
 #   else:
 #       y_pred_LR_str = "Did Not Default"

 #   if y_pred_RFC == 1:
  #      y_pred_RFC_str = "Defaulted"
 #   else:
 #       y_pred_RFC_str = "Did Not Default"

#  if y_pred_DTC == 1:
#        y_pred_DTC_str = "Defaulted"
 #   else:
#        y_pred_DTC_str = "Did Not Default"

#    if y_pred_GBC == 1:
#        y_pred_GBC_str = "Defaulted"
#    else:
#        y_pred_GBC_str = "Did Not Default"

 #   if y_pred_LDA == 1:
 #       y_pred_LDA_str = "Defaulted"
 #   else:
  #      y_pred_LDA_str = "Did Not Default"

  #  return {
   #     'Logistic': y_pred_LR_str,
    #    'Random_Forest': y_pred_RFC_str,
     #   'Decision_Tree': y_pred_DTC_str,
     #   'Gradient_Boosting': y_pred_GBC_str,
      #  'Linear Discriminant': y_pred_LDA_str
   # }

#def get_predictions(X_input):
    # Make predictions using the individual models
    y_pred_LR = final_LR.predict(X_input)[0]
    y_pred_RFC = final_RFC.predict(X_input)[0]
    y_pred_DTC = final_DTC.predict(X_input)[0]
    y_pred_GBC = final_GBC.predict(X_input)[0]
    y_pred_LDA = final_LDA.predict(X_input)[0]

    # Convert the predictions to human-readable format
    y_pred_LR_str = "Defaulted" if y_pred_LR == 1 else "Did Not Default"
    y_pred_RFC_str = "Defaulted" if y_pred_RFC == 1 else "Did Not Default"
    y_pred_DTC_str = "Defaulted" if y_pred_DTC == 1 else "Did Not Default"
    y_pred_GBC_str = "Defaulted" if y_pred_GBC == 1 else "Did Not Default"
    y_pred_LDA_str = "Defaulted" if y_pred_LDA == 1 else "Did Not Default"

    # Get the probability predictions
    y_pred_proba_LR = final_LR.predict_proba(X_input)[0][1]
    y_pred_proba_RFC = final_RFC.predict_proba(X_input)[0][1]
    y_pred_proba_DTC = final_DTC.predict_proba(X_input)[0][1]
    y_pred_proba_GBC = final_GBC.predict_proba(X_input)[0][1]
    y_pred_proba_LDA = final_LDA.predict_proba(X_input)[0][1]

    return {
        'Logistic': {
            'status': y_pred_LR_str,
            'probability': y_pred_proba_LR
        },
        'Random_Forest': {
            'status': y_pred_RFC_str,
            'probability': y_pred_proba_RFC
        },
        'Decision_Tree': {
            'status': y_pred_DTC_str,
            'probability': y_pred_proba_DTC
        },
        'Gradient_Boosting': {
            'status': y_pred_GBC_str,
            'probability': y_pred_proba_GBC
        },
        'Linear Discriminant': {
            'status': y_pred_LDA_str,
            'probability': y_pred_proba_LDA
        }
    }

# 6. Run the FastAPI Application
if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=8000)

##uvicorn app:app --reload


  

    

  

   


													