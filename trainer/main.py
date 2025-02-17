import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import json
import os

# Loading data
X_train = pd.read_csv('train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('train/y_train.txt', delim_whitespace=True, header=None)
X_test = pd.read_csv('test/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('test/y_test.txt', delim_whitespace=True, header=None)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Feature Selection
num_features = 20  # Reduced to 20
selector = SelectKBest(score_func=f_classif, k=num_features)
X_train_selected = selector.fit_transform(X_train, y_train.values.ravel())
X_test_selected = selector.transform(X_test)

selected_feature_indices = selector.get_support(indices=True)

# Loading feature_names
with open('features.txt', 'r') as f:
    feature_names = [line.strip().split(' ')[1] for line in f]  # Extract names
# Get selected feature names, IN ORDER
selected_feature_names = [feature_names[i] for i in selected_feature_indices]

print("Selected features:", selected_feature_names)
print(f"X_train_selected shape: {X_train_selected.shape}")
print(f"X_test_selected shape: {X_test_selected.shape}")

# Training Model
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
model.fit(X_train_selected, y_train.values.ravel())

# Model Evaluation
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X_train_selected, y_train.values.ravel(), cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Saving Model and Feature Info
joblib.dump(model, 'trained_logistic_model.joblib')

# Save BOTH indices and names as JSON (This is the key change)
feature_info = {
    'indices': selected_feature_indices.tolist(),  # Convert to list
    'names': selected_feature_names
}
with open('selected_features.json', 'w') as f:
    json.dump(feature_info, f)

print("Model and selected features (indices and names) saved.")

#verify size
file_size = os.path.getsize('selected_features.json')
print(f"File size: {file_size} bytes")