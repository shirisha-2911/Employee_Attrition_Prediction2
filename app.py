import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# -------------------------------
# Load Saved Model Package
# -------------------------------
@st.cache_resource
def load_model_package():
    return joblib.load("attrition_model.pkl")

data = load_model_package()

deployment_models = data['models']
feature_columns = data['feature_columns']
numeric_features = data['numeric_features']
categorical_features = data['categorical_features']
numeric_defaults = data['numeric_defaults']
categorical_options = data['categorical_options']

# -------------------------------
# Header
# -------------------------------
st.title("Employee Attrition Prediction System")
st.write("Predict whether an employee is likely to leave or stay.")

st.divider()

# -------------------------------
# Sidebar - Model Selection
# -------------------------------
st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    list(deployment_models.keys())
)

selected_model = deployment_models[selected_model_name]

st.sidebar.success(f"Selected Model: {selected_model_name}")

# -------------------------------
# Employee Input Form
# -------------------------------
st.subheader("Employee Information")

input_data = {}

with st.form("employee_prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if feature in numeric_features:

                default_value = numeric_defaults.get(feature, 0)

                input_data[feature] = st.number_input(
                    feature,
                    value=float(default_value)
                )

            elif feature in categorical_features:

                options = categorical_options.get(feature, [])

                if len(options) > 0:

                    input_data[feature] = st.selectbox(
                        feature,
                        options
                    )

                else:

                    input_data[feature] = st.text_input(feature)

            else:

                input_data[feature] = st.text_input(feature)

    predict_button = st.form_submit_button(
        "Predict Attrition"
    )

# -------------------------------
# Prediction
# -------------------------------
if predict_button:

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    prediction = selected_model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    # Display prediction
    if prediction == "Left":

        st.error("Prediction: Employee is likely to LEAVE.")

    else:

        st.success("Prediction: Employee is likely to STAY.")

    # Probability calculation
    if hasattr(selected_model, "predict_proba"):

        probabilities = selected_model.predict_proba(input_df)[0]
        classes = selected_model.classes_

        probability_df = pd.DataFrame({
            "Status": classes,
            "Probability": probabilities
        })

        left_probability = 0

        if "Left" in classes:
            left_probability = probabilities[
                list(classes).index("Left")
            ]

        stay_probability = 1 - left_probability

        col1, col2 = st.columns(2)

        col1.metric(
            "Attrition Probability",
            f"{left_probability:.2%}"
        )

        col2.metric(
            "Retention Probability",
            f"{stay_probability:.2%}"
        )

        st.subheader("Prediction Probability")

        st.bar_chart(
            probability_df.set_index("Status")
        )

        # Risk classification
        if left_probability >= 0.70:
            st.warning("High Attrition Risk")

        elif left_probability >= 0.40:
            st.info("Medium Attrition Risk")

        else:
            st.success("Low Attrition Risk")

    st.caption(f"Model used: {selected_model_name}")
    