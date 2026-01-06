import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/raw/cancer_data.csv")
df.dropna(inplace=True)

# Define features and target
X = df[
    ['Age','Gender','BMI','Smoking','GeneticRisk',
     'PhysicalActivity','AlcoholIntake','CancerHistory']
]
y = df['Diagnosis']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

print("Model training completed")

