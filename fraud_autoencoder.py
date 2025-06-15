# fraud_autoencoder.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 📂 Load the dataset
df = pd.read_csv("creditcard.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# 🔍 Preprocessing
X = df.drop(columns=["Time", "Class"])
y = df["Class"]

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🧪 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🧠 Train AutoEncoder
model = AutoEncoder(contamination=0.001)
model.epochs = 30  # set epochs as an attribute after instantiation

model.fit(X_train)

# 🚨 Make predictions
y_test_scores = model.decision_function(X_test)  # reconstruction error
y_test_pred = model.predict(X_test)              # 0 for inlier, 1 for outlier

# 📊 Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# 📈 Plot error distribution
sns.histplot(y_test_scores, bins=50, kde=True)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.show()
