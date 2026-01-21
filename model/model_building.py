# Wine Cultivar Origin Prediction System - Model Development
# Part A: Model Development

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load the Wine Dataset
print("Loading Wine Dataset...")
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df['cultivar'] = wine_data.target

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:")
print(df.info())

# 2. Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Feature Selection - Selecting 6 features as per requirement
# Selected features based on importance and variety
selected_features = [
    'alcohol',
    'flavanoids',
    'color_intensity',
    'od280/od315_of_diluted_wines',
    'proline',
    'total_phenols'
]

print(f"\nSelected Features: {selected_features}")

# Prepare feature matrix and target vector
X = df[selected_features]
y = df['cultivar']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature Scaling (Mandatory)
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")

# 3. Model Training - Random Forest Classifier
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model training completed!")

# 4. Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification Report
print("\n" + "-"*50)
print("CLASSIFICATION REPORT (Test Set)")
print("-"*50)
print(classification_report(y_test, y_pred_test, 
                          target_names=['Cultivar 0', 'Cultivar 1', 'Cultivar 2']))

# Confusion Matrix
print("\n" + "-"*50)
print("CONFUSION MATRIX (Test Set)")
print("-"*50)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Feature Importance
print("\n" + "-"*50)
print("FEATURE IMPORTANCE")
print("-"*50)
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# 5. Save the Model and Scaler
print("\n" + "="*50)
print("SAVING MODEL AND SCALER")
print("="*50)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
model_path = 'model/wine_cultivar_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Save the scaler
scaler_path = 'model/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Save feature names for reference
feature_names_path = 'model/feature_names.pkl'
joblib.dump(selected_features, feature_names_path)
print(f"Feature names saved to: {feature_names_path}")

print("\n" + "="*50)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("="*50)

# Test prediction with a sample
print("\n" + "="*50)
print("SAMPLE PREDICTION TEST")
print("="*50)

sample_data = X_test.iloc[0:1]
print(f"\nSample input:\n{sample_data}")

sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)
prediction_proba = model.predict_proba(sample_scaled)

print(f"\nPredicted Cultivar: {prediction[0]}")
print(f"Actual Cultivar: {y_test.iloc[0]}")
print(f"\nPrediction Probabilities:")
for i, prob in enumerate(prediction_proba[0]):
    print(f"  Cultivar {i}: {prob:.4f} ({prob*100:.2f}%)")

