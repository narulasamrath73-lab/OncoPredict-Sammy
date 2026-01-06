import pandas as pd
from sklearn.preprocessing import StandardScaler
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("data/raw/cancer_data.csv")

    # Remove missing values
    df.dropna(inplace=True)

    # Define features and target
    X = df[
        [
            'Age',
            'Gender',
            'BMI',
            'Smoking',
            'GeneticRisk',
            'PhysicalActivity',
            'AlcoholIntake',
            'CancerHistory'
        ]
    ]

    y = df['Diagnosis']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

