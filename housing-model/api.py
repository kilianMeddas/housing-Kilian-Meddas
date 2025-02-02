from flask import Flask, render_template_string
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load dataset and remove duplicates
df = pd.read_csv("housing.csv")
df.drop_duplicates(inplace=True)

# Convert 'ocean_proximity' to numerical values
df['ocean_proximity'] = df['ocean_proximity'].map({
    '<1H OCEAN': 1,
    'INLAND': 2,
    'ISLAND': 3,
    'NEAR BAY': 4,
    'NEAR OCEAN': 5
}).fillna(0)

# Split dataset into training and testing sets
data_train, data_test = train_test_split(df, test_size=0.2)

# Train and test with ['median_income', 'housing_median_age', 'total_rooms']
features_columns_multi = ['median_income', 'housing_median_age', 'total_rooms']
target_columns = ['median_house_value']

X_train_multi = data_train[features_columns_multi]
Y_train = data_train[target_columns]
X_test_multi = data_test[features_columns_multi]
Y_test = data_test[target_columns]

linear_regression_model_multi = LinearRegression()
linear_regression_model_multi.fit(X_train_multi, Y_train)

# Predict and evaluate
pred_multi = linear_regression_model_multi.predict(X_test_multi)
MSE_multi = mean_squared_error(Y_test, pred_multi)
R2_multi = r2_score(Y_test, pred_multi)

# Train and test with only 'median_income'
features_columns_single = ['median_income']

X_train_single = data_train[features_columns_single]
X_test_single = data_test[features_columns_single]

linear_regression_model_single = LinearRegression()
linear_regression_model_single.fit(X_train_single, Y_train)

# Predict and evaluate
pred_single = linear_regression_model_single.predict(X_test_single)
MSE_single = mean_squared_error(Y_test, pred_single)
R2_single = r2_score(Y_test, pred_single)

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, pred_multi, label='Multi Features')
plt.scatter(Y_test, pred_single, label='Single Feature (median_income)')
plt.plot(Y_test, Y_test, color='red', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig("output/prediction_vs_actual_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# API endpoint to display results
@app.route('/')
def display_results():
    html = """
    <h1>Model Performance Comparison</h1>
    <h2>Multi-Feature Model (median_income, housing_median_age, total_rooms)</h2>
    <p><strong>MSE:</strong> {{ mse_multi }}</p>
    <p><strong>R²:</strong> {{ r2_multi }}</p>

    <h2>Single-Feature Model (median_income)</h2>
    <p><strong>MSE:</strong> {{ mse_single }}</p>
    <p><strong>R²:</strong> {{ r2_single }}</p>
    """
    return render_template_string(html, mse_multi=MSE_multi, r2_multi=R2_multi, mse_single=MSE_single, r2_single=R2_single)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
