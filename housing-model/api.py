import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('future.no_silent_downcasting', True)
# Create a directory named "output" for saving generated plots and other results.
directory_name = "output"

try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully")
    print("See the result in it")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")
print('\n\n\n')

# Load dataset and remove duplicates
df = pd.read_csv("housing.csv")
df.drop_duplicates(inplace=True)

# Display first 10 rows of the dataset
print(f'Ten head of csv :  \n\n{df.head(10)}\n\n')

# Display descriptive statistics for numerical columns
print(f"Describe of csv : \n\n{df.describe()}\n\n")

# Display dataset information, including data types and missing values
print(f"\n\n {df.info()} : We don't need to use fillna according to the df.info\n\n")

# Display unique values in the 'ocean_proximity' column
print(f"Unique in ocean_proximity : \n\n {df.ocean_proximity.unique()}")

# Convert 'ocean_proximity' to numerical values and join it back to the original DataFrame
df1 = pd.DataFrame(data = {'id_ocean': df.ocean_proximity})
df1 = df1.replace(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY','NEAR OCEAN'], [1,2,3,4,5])
df = df.join(df1)
df = df.fillna(0)


# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Create and save a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.savefig("output/heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n\n\n\n\n See the results in '{directory_name}'")

# Plot and save a histogram of 'median_house_value'
df.median_house_value.hist()
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Histogram of Median House Value")
plt.savefig("output/median_house_value_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# Create and save a boxplot of 'median_house_value' grouped by 'ocean_proximity'
fig = px.box(data_frame = df, y = 'median_house_value', color= 'ocean_proximity')
fig.write_image("output/median_house_value_boxplot.png", width=800, height=600, scale=2)

# Split dataset into training and testing sets
data_train, data_test = train_test_split(df, test_size = 0.2)

print('''
    ############################################################################
    ############################################################################
    ############################################################################
      Train and test with ['median_income', 'housing_median_age', 'total_rooms']\n
    ############################################################################
    ############################################################################
    ############################################################################
    ''')
# Define features and target variables for training and testing
features_columns = ['median_income', 'housing_median_age', 'total_rooms']
target_columns = ['median_house_value']

# features chose among the most "relevant" 69% for median_income, 
# and only 0.11 for housing_median_age and 0.13 for total_rooms
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
MSE_train = mean_squared_error(y_pred = y_train_pred, y_true = Y_train)
MSE_test = mean_squared_error(y_pred = y_test_pred, y_true = Y_test)

R2_train = r2_score(y_pred = y_train_pred, y_true = Y_train)
R2_test = r2_score(y_pred = y_test_pred, y_true = Y_test)

# Display performance metrics
print("\n\n#### Metric performance ####")

print('MSE Train', MSE_train)
print('MSE Test', MSE_test)
print('R2 Train', R2_train)
print('R2 Test', R2_test)


# Transform target variable using cube root transformation for non-linear relationships
# Train a new model using the transformed target variable
y_train_rc = Y_train ** (1/3)
y_test_rc = Y_test ** (1/3)

linear_regression_model_rc = LinearRegression()
linear_regression_model_rc.fit(X_train, y_train_rc)

# Predict on transformed data and inverse the transformation
y_train_pred = linear_regression_model_rc.predict(X_train)**3
y_test_pred = linear_regression_model_rc.predict(X_test)**3


# Evaluate performance of the transformed model
MSE_train = mean_squared_error(y_pred = y_train_pred, y_true = Y_train)
MSE_test = mean_squared_error(y_pred = y_test_pred, y_true = Y_test)

R2_train = r2_score(y_pred = y_train_pred, y_true = Y_train)
R2_test = r2_score(y_pred = y_test_pred, y_true = Y_test)

# Display performance metrics of the transformed model
print("\n\n#### Metric performance transformed (Transform target variable using cube root transformation for non-linear relationships) ####")
print('MSE Train', MSE_train)
print('MSE Test', MSE_test)
print('R2 Train', R2_train)
print('R2 Test', R2_test)

# Visualize predictions vs. actual values
plt.scatter(Y_test, linear_regression_model_rc.predict(X_test)**3, label = 'Root Cube')
plt.scatter(Y_test, linear_regression_model.predict(X_test), label = 'Baseline')
plt.scatter(Y_test, Y_test, label = 'Perfect')
plt.legend()
plt.savefig("output/prediction_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()

print('''\n\n
    ############################################################################
    ############################################################################
    ############################################################################
      Train and test with only 'median_income'\n
    ############################################################################
    ############################################################################
    ############################################################################
    ''')
# Define features and target variables for training and testing
features_columns = ['median_income']
target_columns = ['median_house_value']

# features chose among the most "relevant" 69% for median_income, 
# and only 0.11 for housing_median_age and 0.13 for total_rooms
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
MSE_train = mean_squared_error(y_pred = y_train_pred, y_true = Y_train)
MSE_test = mean_squared_error(y_pred = y_test_pred, y_true = Y_test)

R2_train = r2_score(y_pred = y_train_pred, y_true = Y_train)
R2_test = r2_score(y_pred = y_test_pred, y_true = Y_test)

print("\n\n#### Metric performance ####")
# Display performance metrics
print('MSE Train', MSE_train)
print('MSE Test', MSE_test)
print('R2 Train', R2_train)
print('R2 Test', R2_test)


# Transform target variable using cube root transformation for non-linear relationships
# Train a new model using the transformed target variable
y_train_rc = Y_train ** (1/3)
y_test_rc = Y_test ** (1/3)

linear_regression_model_rc = LinearRegression()
linear_regression_model_rc.fit(X_train, y_train_rc)

# Predict on transformed data and inverse the transformation
y_train_pred = linear_regression_model_rc.predict(X_train)**3
y_test_pred = linear_regression_model_rc.predict(X_test)**3


# Evaluate performance of the transformed model
MSE_train = mean_squared_error(y_pred = y_train_pred, y_true = Y_train)
MSE_test = mean_squared_error(y_pred = y_test_pred, y_true = Y_test)

R2_train = r2_score(y_pred = y_train_pred, y_true = Y_train)
R2_test = r2_score(y_pred = y_test_pred, y_true = Y_test)

# Display performance metrics of the transformed model
print("\n\n#### Metric performance transformed (Transform target variable using cube root transformation for non-linear relationships) ####")
print('MSE Train', MSE_train)
print('MSE Test', MSE_test)
print('R2 Train', R2_train)
print('R2 Test', R2_test)

# Visualize predictions vs. actual values
plt.scatter(Y_test, linear_regression_model_rc.predict(X_test)**3, label = 'Root Cube')
plt.scatter(Y_test, linear_regression_model.predict(X_test), label = 'Baseline')
plt.scatter(Y_test, Y_test, label = 'Perfect')
plt.legend()
plt.savefig("output/prediction_vs_actual_one_feat.png", dpi=300, bbox_inches="tight")