from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

# Load the trained ML model from MLflow
model = mlflow.pyfunc.load_model("models:/HousingModel/Production")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "median_income" not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Convert input to model format
    input_data = [[data["median_income"]]]
    prediction = model.predict(input_data)

    return jsonify({"median_house_value": prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
