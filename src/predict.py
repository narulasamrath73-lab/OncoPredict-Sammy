## IMPORT LIBRARIES
import pandas as pd
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

## LOADING DATASET
df = pd.read_csv("C:/Users/Samrath Narula/Downloads/The_Cancer_data_1500_V2.csv")  

## Remove rows with missing values
df.dropna(inplace=True)

## DEFINE FEATURES (X) AND TARGET (y)
X = df[
    [
        'Age',
        'Gender',
        'BMI',
        'Smoking',
        'GeneticRisk',
        'PhysicalActivity',
        'AlcoholIntake',
        'CancerHistory']]

y = df['Diagnosis'] 

##  FEATURE SCALING
scaler = StandardScaler()             
X_scaled = scaler.fit_transform(X)   

##  TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


## TRAIN LOGISTIC REGRESSION MODEL
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

## PREDICTION ON TEST DATA
y_pred = model.predict(X_test)

## EVALUATION
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

def predict_from_gui():
    try:
        
        user_input = pd.DataFrame([{
            'Age': int(age.get()),
            'Gender': int(gender.get()),
            'BMI': float(bmi.get()),
            'Smoking': int(smoking.get()),
            'GeneticRisk': int(genetic.get()),
            'PhysicalActivity': float(activity.get()),
            'AlcoholIntake': float(alcohol.get()),
            'CancerHistory': int(history.get())  }])

     
        user_scaled = scaler.transform(user_input)

        ## Predict using trained model
        prediction = model.predict(user_scaled)
        probability = model.predict_proba(user_scaled)[0][1] * 100


        if prediction[0] == 1:
            result = f"CANCER DETECTED\nRisk: {round(probability)}%"

        else:
            result = f" CANCER DETECTED\nRisk: {round(probability)}%"


        messagebox.showinfo("Prediction Result", result)

    except:
        ## Error handling for wrong input
        messagebox.showerror("Error", "Please enter valid numeric values")

## USER INPUT FROM TERMINAL 
print("Enter Patient Details for Prediction")

age = int(input("Age: "))
gender = int(input("Gender (0 = Female, 1 = Male): "))
bmi = float(input("BMI: "))
smoking = int(input("Smoking (0 = No, 1 = Yes): "))
genetic = int(input("Genetic Risk (0 = No, 1 = Yes): "))
activity = float(input("Physical Activity (hours/week): "))
alcohol = int(input("Alcohol Intake (0 = No, 1 = Yes): "))
history = int(input("Family Cancer History (0 = No, 1 = Yes): "))

## Create DataFrame
user_input = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Smoking': smoking,
    'GeneticRisk': genetic,
    'PhysicalActivity': activity,
    'AlcoholIntake': alcohol,
    'CancerHistory': history}])


user_scaled = scaler.transform(user_input)

## Predict
prediction = model.predict(user_scaled)
probability = model.predict_proba(user_scaled)[0][1] * 100

## Display Result
print(" PREDICTION OF CANCER/ RESULT")


if prediction[0] == 1:
    print("there is chances of CANCER ")
    print(f"Risk Probability: {round(probability, 2)}%")
else:
    print("No chnaces of cancer")
    print(f"Risk Probability: {round(probability, 2)}%")
