import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import tempfile
import os
from mlflow.models import infer_signature
import streamlit as st


# Set MLflow tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Housing Model Experiment")
        
# Declaration variable for file
st.session_state.setdefault('csv', False)
st.session_state.setdefault('uploaded_files', None)  # Initialize with None

# File uploader (Only show if the file is not yet uploaded)
if not st.session_state.csv:
    uploaded_file = st.file_uploader("Upload the housing.csv file", accept_multiple_files=False, type=['csv'])

    if uploaded_file is not None:
        if uploaded_file.name == "housing.csv":
            st.session_state.uploaded_files = uploaded_file
            st.session_state.csv = True
        else:
            st.error("Please upload a file named 'housing.csv'.")

if st.session_state.csv:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to the temporary directory
        file_path = os.path.join(temp_dir, st.session_state.uploaded_files.name)
        with open(file_path, 'wb') as f:
            f.write(st.session_state.uploaded_files.getbuffer())

        # Suppress pandas warnings
        pd.set_option('mode.chained_assignment', None)

        # Load dataset and remove duplicates
        df = pd.read_csv(file_path)  # Load the CSV from the temporary file path
        df.drop_duplicates(inplace=True)

        # Streamlit UI
        st.title("Housing Price Prediction Model")
        st.subheader("Dataset Overview")
        st.dataframe(df)

        # Show dataset summary
        if st.checkbox("Show Dataframe Summary"):
            st.write(df.describe())

        # Show data types
        if st.checkbox("Show Data Types"):
            buffer = pd.DataFrame(df.dtypes).reset_index()
            buffer.columns = ['Column', 'Data Type']
            st.write(buffer)

        # Convert 'ocean_proximity' to numerical values
        ocean_proximity_mapping = {
            '<1H OCEAN': 1, 'INLAND': 2, 'ISLAND': 3, 'NEAR BAY': 4, 'NEAR OCEAN': 5
        }
        df['id_ocean'] = df['ocean_proximity'].map(ocean_proximity_mapping)
        df = df.fillna(0)

        # Calculate the correlation matrix
        corr_matrix = df.corr(numeric_only=True)

        # Create and display a heatmap of the correlation matrix
        if st.checkbox("Show Correlation Heatmap"):
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)

        # Plot and display a histogram of 'median_house_value'
        if st.checkbox("Show Histogram of Median House Value"):
            plt.figure()
            df['median_house_value'].hist()
            plt.xlabel("Median House Value")
            plt.ylabel("Frequency")
            plt.title("Histogram of Median House Value")
            st.pyplot(plt)

        # Split dataset into training and testing sets
        data_train, data_test = train_test_split(
            df, test_size=0.2, random_state=42
        )

        # Define features and target variables for training and testing
        features_columns = ['median_income']
        target_columns = ['median_house_value']

        X_train = data_train[features_columns]
        Y_train = data_train[target_columns]
        X_test = data_test[features_columns]
        Y_test = data_test[target_columns]

        # Train a linear regression model
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(X_train, Y_train)

        # Predict on training and testing sets
        y_train_pred = linear_regression_model.predict(X_train)
        y_test_pred = linear_regression_model.predict(X_test)

        # Evaluate model performance using MSE and R²
        MSE_train = mean_squared_error(Y_train, y_train_pred)
        MSE_test = mean_squared_error(Y_test, y_test_pred)
        R2_train = r2_score(Y_train, y_train_pred)
        R2_test = r2_score(Y_test, y_test_pred)

        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        st.write("MSE Train:", MSE_train)
        st.write("MSE Test:", MSE_test)
        st.write("R2 Train:", R2_train)
        st.write("R2 Test:", R2_test)

        # Visualize predictions vs. actual values
        if st.checkbox("Show Predictions vs. Actual Values"):
            plt.figure()
            plt.scatter(Y_test, y_test_pred, label='Predictions', alpha=0.6)
            plt.plot(Y_test, Y_test, label='Actual', color='red', linestyle='--')
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predictions vs. Actual")
            plt.legend()
            st.pyplot(plt)


        # Log the model and metrics in an MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state_split", 42)
            mlflow.log_param("model_type", "LinearRegression")

            # Log metrics
            metrics = {
                "MSE Train": MSE_train,
                "MSE Test": MSE_test,
                "R2 Train": R2_train,
                "R2 Test": R2_test,
            }
            mlflow.log_metrics(metrics)

            # Set a tag for the run
            mlflow.set_tag("Training Info", "Linear Regression")

            # Infer the model signature
            signature = infer_signature(X_train, linear_regression_model.predict(X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=linear_regression_model,
                artifact_path="housing_model",
                signature=signature,
                input_example=X_train,
                registered_model_name="HousingModel"
            )

        # Now load the model after it has been logged
        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

        # Predict using the loaded model
        predictions = loaded_model.predict(X_test)

        # Save the results
        result = pd.DataFrame(X_test, columns=features_columns)
        result["actual_class"] = Y_test.values.flatten()
        result["predicted_class"] = predictions.flatten()
        st.subheader("Prediction Results")
        st.dataframe(result.head())
