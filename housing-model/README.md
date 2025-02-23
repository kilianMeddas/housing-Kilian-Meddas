# Housing Model

## Description
This project is a machine learning model that predicts housing prices based on the dataset `housing.csv`. The model is built using Python, trained with Linear Regression, and logs its performance using MLflow. Additionally, a Streamlit interface is provided for visualization and interaction.

## Folder Structure
```
- housing-model/
  - Dockerfile
  - api_ml.py
  - housing.csv
  - requirements.txt
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Docker (if running in a container)
- Python 3.11
- Pip
- Virtual Environment (optional but recommended)

### Local Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd housing-model
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run api_ml.py
   ```

## Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t housing-model .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 housing-model
   ```

## Usage
1. Upload the `housing.csv` file in the Streamlit web interface.
2. Explore the dataset using various visualization options.
3. Train and evaluate the linear regression model.
4. View model performance metrics and prediction results.
5. Track experiments and model versions with MLflow.
6. To view the results in MLflow, start the MLflow tracking server:
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```
   Then, open your browser and go to `http://localhost:5000` to explore the logged metrics and model details.

## Files
### `Dockerfile`
Defines the environment setup for the Docker container, including dependencies installation and execution of `api.py`.

### `api_ml.py`
Main Python script that:
- Loads and preprocesses the dataset
- Visualizes correlations and distributions
- Trains a Linear Regression model
- Logs the model with MLflow
- Provides a Streamlit UI for interaction

### `housing.csv`
Dataset containing housing price data used for training the model.

### `requirements.txt`
Contains the required dependencies for the project.

## Authors
- Meddas Kilian


