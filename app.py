import streamlit as st
import numpy as np
import joblib


# ===============================================================
# LOAD TRAINED MODEL AND SCALER
# ===============================================================

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl")


# ===============================================================
# PAGE CONFIGURATION
# ===============================================================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# ===============================================================
# TITLE
# ===============================================================

st.title("🩺 Breast Cancer Prediction")

st.write(
    "Enter the tumor measurements below to predict whether "
    "the tumor is Malignant or Benign."
)


# ===============================================================
# INPUT SECTION
# ===============================================================

st.header("Enter Tumor Information")


# ---------------------------------------------------------------
# MEAN FEATURES
# ---------------------------------------------------------------

st.subheader("Mean Features")

col1, col2, col3 = st.columns(3)

with col1:

    mean_radius = st.number_input(
        "Mean Radius",
        value=0.0
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        value=0.0
    )

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        value=0.0
    )

    mean_concavity = st.number_input(
        "Mean Concavity",
        value=0.0
    )


with col2:

    mean_texture = st.number_input(
        "Mean Texture",
        value=0.0
    )

    mean_area = st.number_input(
        "Mean Area",
        value=0.0
    )

    mean_compactness = st.number_input(
        "Mean Compactness",
        value=0.0
    )

    mean_concave_points = st.number_input(
        "Mean Concave Points",
        value=0.0
    )


with col3:

    mean_symmetry = st.number_input(
        "Mean Symmetry",
        value=0.0
    )

    mean_fractal_dimension = st.number_input(
        "Mean Fractal Dimension",
        value=0.0
    )


# ---------------------------------------------------------------
# ERROR FEATURES
# ---------------------------------------------------------------

st.subheader("Error Features")

col1, col2, col3 = st.columns(3)

with col1:

    radius_error = st.number_input(
        "Radius Error",
        value=0.0
    )

    perimeter_error = st.number_input(
        "Perimeter Error",
        value=0.0
    )

    smoothness_error = st.number_input(
        "Smoothness Error",
        value=0.0
    )

    concavity_error = st.number_input(
        "Concavity Error",
        value=0.0
    )


with col2:

    texture_error = st.number_input(
        "Texture Error",
        value=0.0
    )

    area_error = st.number_input(
        "Area Error",
        value=0.0
    )

    compactness_error = st.number_input(
        "Compactness Error",
        value=0.0
    )

    concave_points_error = st.number_input(
        "Concave Points Error",
        value=0.0
    )


with col3:

    symmetry_error = st.number_input(
        "Symmetry Error",
        value=0.0
    )

    fractal_dimension_error = st.number_input(
        "Fractal Dimension Error",
        value=0.0
    )


# ---------------------------------------------------------------
# WORST FEATURES
# ---------------------------------------------------------------

st.subheader("Worst Features")

col1, col2, col3 = st.columns(3)

with col1:

    worst_radius = st.number_input(
        "Worst Radius",
        value=0.0
    )

    worst_perimeter = st.number_input(
        "Worst Perimeter",
        value=0.0
    )

    worst_smoothness = st.number_input(
        "Worst Smoothness",
        value=0.0
    )

    worst_concavity = st.number_input(
        "Worst Concavity",
        value=0.0
    )


with col2:

    worst_texture = st.number_input(
        "Worst Texture",
        value=0.0
    )

    worst_area = st.number_input(
        "Worst Area",
        value=0.0
    )

    worst_compactness = st.number_input(
        "Worst Compactness",
        value=0.0
    )

    worst_concave_points = st.number_input(
        "Worst Concave Points",
        value=0.0
    )


with col3:

    worst_symmetry = st.number_input(
        "Worst Symmetry",
        value=0.0
    )

    worst_fractal_dimension = st.number_input(
        "Worst Fractal Dimension",
        value=0.0
    )


# ===============================================================
# PREDICTION
# ===============================================================

if st.button(
    "🔍 PREDICT",
    use_container_width=True
):

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


    # -----------------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------------

    new_data_scaled = scaler.transform(
        new_data
    )


    # -----------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------

    prediction = model.predict(
        new_data_scaled
    )


    # -----------------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------------

    st.header("Prediction Result")


    if prediction[0] == 0:

        st.error(
            "Prediction: MALIGNANT"
        )

    else:

        st.success(
            "Prediction: BENIGN"
        )


# ===============================================================
# FOOTER
# ===============================================================

st.divider()

st.caption(
    "Machine Learning Model: Logistic Regression"
)