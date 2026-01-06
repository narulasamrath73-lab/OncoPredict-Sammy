import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/raw/cancer_data.csv")
df.dropna(inplace=True)

X = df[
    ['Age','Gender','BMI','Smoking','GeneticRisk',
     'PhysicalActivity','AlcoholIntake','CancerHistory']]
y = df['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
cnm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cnm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Cross validation
cv = StratifiedKFold(n_splits=9, shuffle=True, random_state=19)
cvs = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

print("Cross Validation Scores:", cvs)
print("Mean of  CV Accuracy:", cv_scores.mean())

