# ===============================================================
# BREAST CANCER PREDICTION - CLIENT PROGRAM
# ===============================================================

import numpy as np
import joblib


# ===============================================================
# 1. LOAD TRAINED MODEL
# ===============================================================

model = joblib.load(
    "breast_cancer_model.pkl"
)

scaler = joblib.load(
    "breast_cancer_scaler.pkl"
)


print("=" * 70)
print("              BREAST CANCER PREDICTION")
print("=" * 70)

print("\nModel Loaded Successfully")


# ===============================================================
# 2. TAKE USER INPUT
# ===============================================================

print("\nEnter Tumor Information")
print("Enter the following 30 feature values:")


mean_radius = float(
    input("Mean Radius: ")
)

mean_texture = float(
    input("Mean Texture: ")
)

mean_perimeter = float(
    input("Mean Perimeter: ")
)

mean_area = float(
    input("Mean Area: ")
)

mean_smoothness = float(
    input("Mean Smoothness: ")
)

mean_compactness = float(
    input("Mean Compactness: ")
)

mean_concavity = float(
    input("Mean Concavity: ")
)

mean_concave_points = float(
    input("Mean Concave Points: ")
)

mean_symmetry = float(
    input("Mean Symmetry: ")
)

mean_fractal_dimension = float(
    input("Mean Fractal Dimension: ")
)


radius_error = float(
    input("Radius Error: ")
)

texture_error = float(
    input("Texture Error: ")
)

perimeter_error = float(
    input("Perimeter Error: ")
)

area_error = float(
    input("Area Error: ")
)

smoothness_error = float(
    input("Smoothness Error: ")
)

compactness_error = float(
    input("Compactness Error: ")
)

concavity_error = float(
    input("Concavity Error: ")
)

concave_points_error = float(
    input("Concave Points Error: ")
)

symmetry_error = float(
    input("Symmetry Error: ")
)

fractal_dimension_error = float(
    input("Fractal Dimension Error: ")
)


worst_radius = float(
    input("Worst Radius: ")
)

worst_texture = float(
    input("Worst Texture: ")
)

worst_perimeter = float(
    input("Worst Perimeter: ")
)

worst_area = float(
    input("Worst Area: ")
)

worst_smoothness = float(
    input("Worst Smoothness: ")
)

worst_compactness = float(
    input("Worst Compactness: ")
)

worst_concavity = float(
    input("Worst Concavity: ")
)

worst_concave_points = float(
    input("Worst Concave Points: ")
)

worst_symmetry = float(
    input("Worst Symmetry: ")
)

worst_fractal_dimension = float(
    input("Worst Fractal Dimension: ")
)


# ===============================================================
# 3. CREATE INPUT ARRAY
# ===============================================================

new_data = np.array([[
    mean_radius,
    mean_texture,
    mean_perimeter,
    mean_area,
    mean_smoothness,
    mean_compactness,
    mean_concavity,
    mean_concave_points,
    mean_symmetry,
    mean_fractal_dimension,

    radius_error,
    texture_error,
    perimeter_error,
    area_error,
    smoothness_error,
    compactness_error,
    concavity_error,
    concave_points_error,
    symmetry_error,
    fractal_dimension_error,

    worst_radius,
    worst_texture,
    worst_perimeter,
    worst_area,
    worst_smoothness,
    worst_compactness,
    worst_concavity,
    worst_concave_points,
    worst_symmetry,
    worst_fractal_dimension
]])


# ===============================================================
# 4. SCALE NEW DATA
# ===============================================================

new_data_scaled = scaler.transform(
    new_data
)


# ===============================================================
# 5. MAKE PREDICTION
# ===============================================================

prediction = model.predict(
    new_data_scaled
)


# ===============================================================
# 6. DISPLAY RESULT
# ===============================================================

print("\n")
print("=" * 70)
print("                     RESULT")
print("=" * 70)


if prediction[0] == 0:
    print("\nPrediction: MALIGNANT")
    print("Please consult a medical professional.")
else:
    print("\nPrediction: BENIGN")
    print("No malignant pattern detected by the model.")

# ===============================================================
# END OF PROGRAM
# ===============================================================