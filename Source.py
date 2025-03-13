import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to clean sensor data
def clean_sensor_data(df, x_col, y_col, z_col):
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
    return df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce")

# Function to compute magnitude
def compute_magnitude(df):
    return np.sqrt(df.iloc[:, 0]**2 + df.iloc[:, 1]**2 + df.iloc[:, 2]**2)

# Load datasets
train_acc = pd.read_csv("acc train.csv")
train_gyro = pd.read_csv("Gyro train.csv")
train_mag = pd.read_csv("Mag train.csv")

test_acc = pd.read_csv("acc test.csv")
test_gyro = pd.read_csv("gyro test.csv")
test_mag = pd.read_csv("mag test.csv")

# Extract numerical sensor data
train_acc_clean = clean_sensor_data(train_acc, "X (mg)", "Y (mg)", "Z (mg)")
train_gyro_clean = clean_sensor_data(train_gyro, "X (dps)", "Y (dps)", "Z (dps)")
train_mag_clean = clean_sensor_data(train_mag, "X (mGa)", "Y (mGa)", "Z (mGa)")

test_acc_clean = clean_sensor_data(test_acc, "X (mg)", "Y (mg)", "Z (mg)")
test_gyro_clean = clean_sensor_data(test_gyro, "X (dps)", "Y (dps)", "Z (dps)")
test_mag_clean = clean_sensor_data(test_mag, "X (mGa)", "Y (mGa)", "Z (mGa)")

# Compute Stability Metrics (Magnitudes)
train_acc_clean["Magnitude"] = compute_magnitude(train_acc_clean)
train_gyro_clean["Magnitude"] = compute_magnitude(train_gyro_clean)
train_mag_clean["Magnitude"] = compute_magnitude(train_mag_clean)

test_acc_clean["Magnitude"] = compute_magnitude(test_acc_clean)
test_gyro_clean["Magnitude"] = compute_magnitude(test_gyro_clean)
test_mag_clean["Magnitude"] = compute_magnitude(test_mag_clean)

# Combine only relevant features for ML model (no rolling variance)
X_train = pd.concat([train_acc_clean["Magnitude"], train_gyro_clean["Magnitude"], train_mag_clean["Magnitude"]], axis=1)
X_test = pd.concat([test_acc_clean["Magnitude"], test_gyro_clean["Magnitude"], test_mag_clean["Magnitude"]], axis=1)

# Generate synthetic labels assuming binary classification
# 0 = Stable, 1 = Unstable
y_train = [0] * (len(X_train) // 2) + [1] * (len(X_train) - len(X_train) // 2)
y_test = [0] * (len(X_test) // 2) + [1] * (len(X_test) - len(X_test) // 2)

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=20)
}

# Train models and evaluate accuracy
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    results[name] = {"Training Accuracy": train_accuracy, "Testing Accuracy": test_accuracy}

# Print model accuracy results
print("\nOptimized Model Accuracy Results:")
for model, scores in results.items():
    print(f"{model}: Training Accuracy = {scores['Training Accuracy']:.2%}, Testing Accuracy = {scores['Testing Accuracy']:.2%}")
