import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("adult 3.csv")
data = data.dropna()

# Encode target
label_enc = LabelEncoder()
data["income"] = label_enc.fit_transform(data["income"])

# One-hot encode features
X = pd.get_dummies(data.drop("income", axis=1))
y = data["income"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Prepare TabNet data types
y_train = y_train.values.astype(np.int64)
y_test = y_test.values.astype(np.int64)

# Train TabNet
tabnet = TabNetClassifier(verbose=0)
tabnet.fit(X_train_scaled, y_train, max_epochs=100)

# Predict and evaluate
y_pred = tabnet.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", cm)

# Plot accuracy bar
plt.bar(["TabNet"], [accuracy], color="skyblue")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.text(0, accuracy + 0.01, f"{accuracy:.2f}", ha='center')
plt.tight_layout()
plt.savefig("accuracy_plot.png")
plt.show()
