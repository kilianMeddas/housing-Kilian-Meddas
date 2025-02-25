import os
import time
import warnings
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)
warnings.filterwarnings("ignore")

# Set the MLflow tracking URI from the environment variable
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
mlflow.set_tracking_uri(mlflow_tracking_uri)

def wait_for_model():
    """Wait until the model is available in the Production stage."""
    while True:
        try:
            mlflow.pyfunc.load_model("models:/HousingModel/Production")
            print("Model is available in Production.")
            break
        except Exception:
            print("Model not available yet, retrying in 5 seconds...")
            time.sleep(5)

wait_for_model()

# Load the model from the Production stage
model = mlflow.pyfunc.load_model("models:/HousingModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({'predicted_median_house_value': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
