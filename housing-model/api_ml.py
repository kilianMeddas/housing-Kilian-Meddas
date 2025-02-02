import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature

# Suppress pandas warnings
pd.set_option('mode.chained_assignment', None)

params = {
    "test_size": 0.2,                # Proportion of data used for testing
    "random_state_split": 42,        # Random seed for train-test split
    "model_type": "LinearRegression",# Model type being used
}


# Load dataset and remove duplicates
df = pd.read_csv("housing.csv")
df.drop_duplicates(inplace=True)

# Dataset overview
head = df.head(10)
describe = df.describe()
df.info()

# Convert 'ocean_proximity' to numerical values
ocean_proximity_mapping = {
    '<1H OCEAN': 1, 'INLAND': 2, 'ISLAND': 3, 'NEAR BAY': 4, 'NEAR OCEAN': 5
}
df['id_ocean'] = df['ocean_proximity'].map(ocean_proximity_mapping)
df = df.fillna(0)

# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Create and save a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.savefig("correlation_heatmap.png")
plt.close()

# Plot and save a histogram of 'median_house_value'
df['median_house_value'].hist()
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Histogram of Median House Value")
plt.savefig("median_house_value_histogram.png")
plt.close()

# Create and save a boxplot of 'median_house_value' grouped by 'ocean_proximity'
fig = px.box(data_frame=df, y='median_house_value', color='ocean_proximity')
fig.write_image("boxplot_median_house_value.png")

# Split dataset into training and testing sets
data_train, data_test = train_test_split(
    df, test_size=params["test_size"], random_state=params["random_state_split"]
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

print("\n\n#### Metric performance ####")
print('MSE Train:', MSE_train)
print('MSE Test:', MSE_test)
print('R2 Train:', R2_train)
print('R2 Test:', R2_test)

# Transform target variable using cube root transformation for non-linear relationships
y_train_rc = Y_train ** (1 / 3)
y_test_rc = Y_test ** (1 / 3)

linear_regression_model_rc = LinearRegression()
linear_regression_model_rc.fit(X_train, y_train_rc)

# Predict on transformed data and inverse the transformation
y_train_pred_rc = linear_regression_model_rc.predict(X_train) ** 3
y_test_pred_rc = linear_regression_model_rc.predict(X_test) ** 3

# Evaluate performance of the transformed model
MSE_train_rc = mean_squared_error(Y_train, y_train_pred_rc)
MSE_test_rc = mean_squared_error(Y_test, y_test_pred_rc)

R2_train_rc = r2_score(Y_train, y_train_pred_rc)
R2_test_rc = r2_score(Y_test, y_test_pred_rc)

print("\n\n#### Metric performance transformed (cube root) ####")
print('MSE Train (RC):', MSE_train_rc)
print('MSE Test (RC):', MSE_test_rc)
print('R2 Train (RC):', R2_train_rc)
print('R2 Test (RC):', R2_test_rc)

# Visualize predictions vs. actual values
plt.scatter(Y_test, y_test_pred_rc, label='Root Cube', alpha=0.6)
plt.scatter(Y_test, y_test_pred, label='Baseline', alpha=0.6)
plt.plot(Y_test, Y_test, label='Perfect', color='red', linestyle='--')
plt.legend()
plt.title("Predictions vs. Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("predictions_vs_actual.png")
plt.close()

# Set MLflow tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Housing Model Experiment")

# Log the model and metrics in an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)

    # Log metrics
    metrics = {
        "MSE Train": MSE_train_rc,
        "MSE Test": MSE_test_rc,
        "R2 Train": R2_train_rc,
        "R2 Test": R2_test_rc,
    }
    mlflow.log_metrics(metrics)

    # Set a tag for the run
    mlflow.set_tag("Training Info", "Linear Regression with cube root transformation")

    # Infer the model signature
    signature = infer_signature(X_train, linear_regression_model_rc.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=linear_regression_model_rc,
        artifact_path="housing_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="HousingModel"
    )

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# Predict using the loaded model
predictions = loaded_model.predict(X_test)

# Save the results
result = pd.DataFrame(X_test, columns=features_columns)
result["actual_class"] = Y_test.values.flatten()
result["predicted_class"] = predictions.flatten()
print(result.head())
