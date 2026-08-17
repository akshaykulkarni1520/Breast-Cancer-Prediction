# ===============================================================
# BREAST CANCER PREDICTION - MODEL TRAINING
# ===============================================================

# ===============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ===============================================================
# 2. LOAD DATASET
# ===============================================================

data = load_breast_cancer()

print("=" * 70)
print("              BREAST CANCER PREDICTION")
print("=" * 70)

print("\nDataset Loaded Successfully")


# ===============================================================
# 3. EXPLORE DATASET
# ===============================================================

print("\n")
print("=" * 70)
print("                 DATASET INFORMATION")
print("=" * 70)

print("\nNumber of Records :", data.data.shape[0])
print("Number of Features:", data.data.shape[1])

print("\nFeature Names:")
print(data.feature_names)

print("\nTarget Names:")
print(data.target_names)

print("\nTarget Mapping:")
print("0 -> Malignant")
print("1 -> Benign")


# ===============================================================
# 4. CREATE DATAFRAME
# ===============================================================

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Target"] = data.target


print("\n")
print("=" * 70)
print("                  FIRST 5 RECORDS")
print("=" * 70)

print(df.head())


# ===============================================================
# 5. DATASET INFORMATION
# ===============================================================

print("\n")
print("=" * 70)
print("                   DATASET INFO")
print("=" * 70)

df.info()


# ===============================================================
# 6. CHECK MISSING VALUES
# ===============================================================

print("\n")
print("=" * 70)
print("                   MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ===============================================================
# 7. SUMMARY STATISTICS
# ===============================================================

print("\n")
print("=" * 70)
print("                 SUMMARY STATISTICS")
print("=" * 70)

print(df.describe())


# ===============================================================
# 8. EXPLORATORY DATA ANALYSIS
# ===============================================================

print("\n")
print("=" * 70)
print("             EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------------
# 8.1 TARGET DISTRIBUTION
# ---------------------------------------------------------------

print("\nTarget Distribution:")

print(df["Target"].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Target",
    data=df
)

plt.title("Target Distribution")
plt.xlabel("Tumor Type")
plt.ylabel("Number of Records")

plt.xticks(
    [0, 1],
    ["Malignant", "Benign"]
)

plt.show()


# ---------------------------------------------------------------
# 8.2 CORRELATION MATRIX
# ---------------------------------------------------------------

plt.figure(figsize=(14, 10))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Feature Correlation Matrix")

plt.show()


# ===============================================================
# 9. SEPARATE FEATURES AND TARGET
# ===============================================================

X = df.drop(
    "Target",
    axis=1
)

y = df["Target"]


print("\n")
print("=" * 70)
print("                FEATURES AND TARGET")
print("=" * 70)

print("\nFeatures Shape :", X.shape)
print("Target Shape   :", y.shape)


# ===============================================================
# 10. TRAIN-TEST SPLIT
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n")
print("=" * 70)
print("                  TRAIN TEST SPLIT")
print("=" * 70)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])


# ===============================================================
# 11. FEATURE SCALING
# ===============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


print("\n")
print("=" * 70)
print("                  FEATURE SCALING")
print("=" * 70)

print("\nFeature Scaling Completed Successfully")


# ===============================================================
# 12. BUILD MACHINE LEARNING MODEL
# ===============================================================

model = LogisticRegression(
    max_iter=1000
)


print("\n")
print("=" * 70)
print("               MACHINE LEARNING MODEL")
print("=" * 70)

print("\nModel: Logistic Regression")


# ===============================================================
# 13. TRAIN THE MODEL
# ===============================================================

model.fit(
    X_train_scaled,
    y_train
)


print("\nModel Training Completed Successfully")


# ===============================================================
# 14. MAKE PREDICTIONS
# ===============================================================

y_pred = model.predict(
    X_test_scaled
)


# ===============================================================
# 15. MODEL EVALUATION
# ===============================================================


# ---------------------------------------------------------------
# 15.1 ACCURACY
# ---------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("                     ACCURACY")
print("=" * 70)

print("\nAccuracy:", accuracy)

print("Accuracy Percentage:",
      accuracy * 100, "%")


# ---------------------------------------------------------------
# 15.2 CONFUSION MATRIX
# ---------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("                  CONFUSION MATRIX")
print("=" * 70)

print("\n", cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ---------------------------------------------------------------
# 15.3 PRECISION
# ---------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("                     PRECISION")
print("=" * 70)

print("\nPrecision:", precision)


# ---------------------------------------------------------------
# 15.4 RECALL
# ---------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("                      RECALL")
print("=" * 70)

print("\nRecall:", recall)


# ---------------------------------------------------------------
# 15.5 F1-SCORE
# ---------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 70)
print("                     F1-SCORE")
print("=" * 70)

print("\nF1-Score:", f1)


# ===============================================================
# 16. CLASSIFICATION REPORT
# ===============================================================

print("\n")
print("=" * 70)
print("                CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Malignant",
            "Benign"
        ]
    )
)


# ===============================================================
# 17. SAVE TRAINED MODEL AND SCALER
# ===============================================================

joblib.dump(
    model,
    "breast_cancer_model.pkl"
)

joblib.dump(
    scaler,
    "breast_cancer_scaler.pkl"
)


print("\n")
print("=" * 70)
print("                 MODEL SAVING")
print("=" * 70)

print("\nTrained Model Saved Successfully")
print("File: breast_cancer_model.pkl")

print("\nScaler Saved Successfully")
print("File: breast_cancer_scaler.pkl")


# ===============================================================
# 18. OBSERVATIONS
# ===============================================================

print("\n")
print("=" * 70)
print("                    OBSERVATIONS")
print("=" * 70)

print("\n1. The dataset contains 569 records.")

print("2. The dataset contains 30 medical features.")

print("3. Missing values were checked successfully.")

print("4. Feature scaling was performed using StandardScaler.")

print("5. The dataset was divided into training and testing sets.")

print("6. Logistic Regression was used as the classification model.")

print("7. The model was evaluated using Accuracy,")
print("   Confusion Matrix, Precision, Recall and F1-Score.")




# ===============================================================
# END OF PROGRAM
# ===============================================================