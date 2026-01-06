import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/raw/cancer_data.csv")
df.dropna(inplace=True)

X = df[
    ['Age','Gender','BMI','Smoking','GeneticRisk',
     'PhysicalActivity','AlcoholIntake','CancerHistory']
]
y = df['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_scaled, y)

print("Enter Patient Details")

age = int(input("Age: "))
gender = int(input("Gender (0=Female,1=Male): "))
bmi = float(input("BMI: "))
smoking = int(input("Smoking (0/1): "))
genetic = int(input("Genetic Risk (0/1): "))
activity = float(input("Physical Activity: "))
alcohol = int(input("Alcohol Intake (0/1): "))
history = int(input("Family Cancer History (0/1): "))

user_input = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Smoking': smoking,
    'GeneticRisk': genetic,
    'PhysicalActivity': activity,
    'AlcoholIntake': alcohol,
    'CancerHistory': history
}])

user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)
probability = model.predict_proba(user_scaled)[0][1] * 100

if prediction[0] == 1:
    print("Cancer is being Detected")
else:
    print("No Cancer Detectionn probability is found")

print(f"Risk Probability: {round(probability,2)}%")

