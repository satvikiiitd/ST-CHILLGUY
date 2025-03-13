# ChillGuy: AI-Powered Shipment Stability Monitoring

## Overview
ChillGuy is an AI-driven solution that detects **shipment instability** using sensor data from **accelerometers, gyroscopes, and magnetometers**. It identifies excessive **tilting, vibration, and impact** in real-time, ensuring safe transportation of fragile and high-value goods.

## Requirements
### Hardware:
- **SensorTile.Box Pro** (for data collection)
- Laptop/PC for processing

### Software & Libraries:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (optional for visualization)

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ChillGuy-AI-Logistics.git
   cd ChillGuy-AI-Logistics
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
1. **Collect sensor data using SensorTile.Box Pro.**
2. Save the data in CSV format under the `Dataset/` folder.
3. Ensure files are named correctly:
   - `acc_train.csv`, `gyro_train.csv`, `mag_train.csv`
   - `acc_test.csv`, `gyro_test.csv`, `mag_test.csv`

## Running the Demo
### **1️⃣ Train the Machine Learning Model**
```bash
python Source_Code/train_model.py
```
This trains **Random Forest** and **Logistic Regression** on the collected dataset.

### **2️⃣ Test the Model & Predict Instability**
```bash
python Source_Code/test_model.py
```
This applies the trained model to test data and generates **real-time alerts** for excessive instability.

## Output Interpretation
- If an alert is triggered, it means the shipment is **unstable** and requires corrective action.
- The model provides **accuracy scores** for both training and testing phases.



