
import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# ==========================================
# 2. GOOGLE DRIVE MODEL CONFIGURATION
# ==========================================

MODEL_PATH = "employee_attrition_models.pkl"

FILE_ID = "1vMsbKrn5OpSwBdRlFhVzFMPWVMVur854"


# ==========================================
# 3. DOWNLOAD AND LOAD SAVED MODEL
# ==========================================

@st.cache_resource
def load_model_package():

    try:

        # Download model from Google Drive
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:

            with st.spinner("Downloading machine learning models... Please wait."):

                downloaded_file = gdown.download(
                    id=FILE_ID,
                    output=MODEL_PATH,
                    quiet=False
                )

                if downloaded_file is None:
                    st.error(
                        "Model download failed. Check Google Drive sharing permissions."
                    )
                    st.stop()

        # Check model file
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            st.error("Model file not found or download is incomplete.")
            st.stop()

        # Load model package
        with st.spinner("Loading machine learning models..."):
            data = joblib.load(MODEL_PATH)

        return data

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


data = load_model_package()


# ==========================================
# 4. EXTRACT MODEL INFORMATION
# ==========================================

deployment_models = data["models"]

feature_columns = data["feature_columns"]

numeric_features = data["numeric_features"]

categorical_features = data["categorical_features"]

numeric_defaults = data["numeric_defaults"]

categorical_options = data["categorical_options"]

categorical_defaults = data.get("categorical_defaults", {})

best_model_name = data.get("best_model_name", None)


# ==========================================
# 5. HEADER
# ==========================================

st.title("👥 Employee Attrition Prediction System")

st.write(
    """
    This application predicts whether an employee is likely
    to leave the organization or stay based on employee details.
    """
)

st.divider()


# ==========================================
# 6. SIDEBAR - MODEL SELECTION
# ==========================================

st.sidebar.header("⚙️ Model Selection")

model_names = list(deployment_models.keys())

default_index = (
    model_names.index(best_model_name)
    if best_model_name in model_names
    else 0
)

selected_model_name = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    model_names,
    index=default_index
)

selected_model = deployment_models[selected_model_name]

st.sidebar.success(
    f"Selected Model: {selected_model_name}"
)

if best_model_name:
    st.sidebar.info(
        f"Best Model: {best_model_name}"
    )


# ==========================================
# 7. EMPLOYEE INPUT FORM
# ==========================================

st.subheader("📝 Enter Employee Information")

input_data = {}

with st.form("employee_prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_columns):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            # Numerical Features
            if feature in numeric_features:

                default_value = numeric_defaults.get(feature, 0)

                input_data[feature] = st.number_input(
                    label=feature,
                    value=float(default_value),
                    format="%.2f"
                )

            # Categorical Features
            elif feature in categorical_features:

                options = categorical_options.get(feature, [])

                if len(options) > 0:

                    default_value = categorical_defaults.get(
                        feature,
                        options[0]
                    )

                    if default_value not in options:
                        default_value = options[0]

                    default_index = options.index(default_value)

                    input_data[feature] = st.selectbox(
                        label=feature,
                        options=options,
                        index=default_index
                    )

                else:

                    input_data[feature] = st.text_input(feature)


    predict_button = st.form_submit_button(
        "🔍 Predict Attrition",
        use_container_width=True
    )


# ==========================================
# 8. PREDICTION
# ==========================================

if predict_button:

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:

        prediction = selected_model.predict(input_df)[0]

        st.divider()

        st.subheader("📊 Prediction Result")

        # Display Prediction
        if str(prediction).strip().lower() == "left":

            st.error(
                "⚠️ Prediction: Employee is likely to LEAVE."
            )

        else:

            st.success(
                "✅ Prediction: Employee is likely to STAY."
            )


        # ==========================================
        # 9. PROBABILITY CALCULATION
        # ==========================================

        if hasattr(selected_model, "predict_proba"):

            probabilities = selected_model.predict_proba(input_df)[0]

            classes = selected_model.classes_

            probability_df = pd.DataFrame({
                "Status": classes,
                "Probability": probabilities
            })

            left_probability = 0
            stay_probability = 0

            for class_name, probability in zip(classes, probabilities):

                if str(class_name).strip().lower() == "left":
                    left_probability = probability

                elif str(class_name).strip().lower() in ["stayed", "stay"]:
                    stay_probability = probability


            col1, col2 = st.columns(2)

            col1.metric(
                "Attrition Probability",
                f"{left_probability:.2%}"
            )

            col2.metric(
                "Retention Probability",
                f"{stay_probability:.2%}"
            )


            st.subheader("📈 Prediction Probability")

            st.bar_chart(
                probability_df.set_index("Status")["Probability"]
            )


            # ==========================================
            # 10. ATTRITION RISK CLASSIFICATION
            # ==========================================

            st.subheader("🚦 Attrition Risk Level")

            if left_probability >= 0.70:

                st.error("🔴 High Attrition Risk")

            elif left_probability >= 0.40:

                st.warning("🟠 Medium Attrition Risk")

            else:

                st.success("🟢 Low Attrition Risk")

        else:

            st.info(
                "Probability scores are not available for this model. "
                "The prediction above is based on its classification output."
            )


        # ==========================================
        # 11. MODEL INFORMATION
        # ==========================================

        st.caption(
            f"Model used: {selected_model_name}"
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Employee Attrition Prediction System | "
    "Machine Learning Project"
)
