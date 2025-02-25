import os
import time
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import streamlit as st
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://mlflow:5001")
mlflow.set_experiment("Housing Model Experiment")

st.title("Housing Price Prediction Model")

# File upload handling
st.session_state.setdefault('csv', False)
st.session_state.setdefault('uploaded_file', None)

uploaded_file = st.file_uploader("Upload housing.csv file", type=['csv'])

if uploaded_file is not None:
    if uploaded_file.name == "housing.csv":
        st.session_state.uploaded_file = uploaded_file
        st.session_state.csv = True
    else:
        st.error("Please upload a file named 'housing.csv'.")

if st.session_state.csv:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "housing.csv")
        with open(file_path, 'wb') as f:
            f.write(st.session_state.uploaded_file.getbuffer())

        # Load and prepare the dataset
        df = pd.read_csv(file_path)
        df.drop_duplicates(inplace=True)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if st.checkbox("Show Descriptive Statistics"):
            st.write(df.describe())

        if st.checkbox("Show Data Types"):
            st.write(df.dtypes)

        # Encode the categorical variable 'ocean_proximity'
        ocean_proximity_mapping = {
            '<1H OCEAN': 1,
            'INLAND': 2,
            'ISLAND': 3,
            'NEAR BAY': 4,
            'NEAR OCEAN': 5
        }
        df['ocean_proximity'] = df['ocean_proximity'].map(ocean_proximity_mapping)
        df.fillna(0, inplace=True)

        # Show correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
            st.pyplot(plt)

        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        X_train = train_data[['median_income']]
        Y_train = train_data[['median_house_value']]
        X_test = test_data[['median_income']]
        Y_test = test_data[['median_house_value']]

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate the model
        metrics = {
            "MSE Train": mean_squared_error(Y_train, y_train_pred),
            "MSE Test": mean_squared_error(Y_test, y_test_pred),
            "R2 Train": r2_score(Y_train, y_train_pred),
            "R2 Test": r2_score(Y_test, y_test_pred),
        }

        st.subheader("Model Performance Metrics")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.4f}")

        if st.checkbox("Show Predictions vs. Actual"):
            plt.figure()
            plt.scatter(Y_test, y_test_pred, label="Predictions", alpha=0.6)
            plt.plot(Y_test, Y_test, color='red', linestyle='--', label="Actual")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predictions vs. Actual Values")
            plt.legend()
            st.pyplot(plt)

        # Log the model with MLflow
        with mlflow.start_run() as run:
            mlflow.log_params({"test_size": 0.2, "random_state": 42, "model": "LinearRegression"})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("Training Info", "Linear Regression Model")

            signature = infer_signature(X_train, model.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="housing_model",
                signature=signature,
                input_example=X_train
            )

            st.write(f"Model saved at: {model_info.model_uri}")

            # Explicitly register the model to obtain its version
            registered_model = mlflow.register_model(model_info.model_uri, "HousingModel")

            # Small pause to ensure registration is complete
            time.sleep(2)

            client = MlflowClient()
            client.transition_model_version_stage(
                name="HousingModel",
                version=registered_model.version,
                stage="Production",
                archive_existing_versions=True
            )
            st.write(f"Model 'HousingModel' version {registered_model.version} transitioned to Production.")

        # Load and test the logged model
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)

        result = pd.DataFrame(X_test, columns=['median_income'])
        result["Actual"] = Y_test.values.flatten()
        result["Predicted"] = predictions.flatten()

        st.subheader("Prediction Results")
        st.dataframe(result.head())
