# Flask and PostgreSQL Integration with Python

## Overview

This project demonstrates how to integrate a Python Flask web application with a PostgreSQL database using the `psycopg2` library. The application provides a simple API for managing house data in a `houses` table within the PostgreSQL database.

---

## Prerequisites

### Tools and Libraries

- Python
- PostgreSQL
- Flask
- psycopg2

### Setup

Before running the application, ensure you have the following:

1. Python virtual environment (`venv`) activated.
2. PostgreSQL server running.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone git@github.com:kilianMeddas/housing-Kilian-Meddas.git
cd housing-Kilian-Meddas
```

### Step 2: Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure PostgreSQL

Update the PostgreSQL credentials in the `app.py` file:

```python
conn = psycopg.connect(database="postgres", user="postgres", password="your_password", host="127.0.0.1", port="5432")
```

Ensure the `postgres` database exists, and the user has the necessary permissions to create additional databases and tables.

---

## Running the Application

1. Start the PostgreSQL server.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Access the API at `http://127.0.0.1:5000`.

---

## API Endpoints

### GET `/houses`

Retrieve all house entries from the `houses` table.

#### Response Exemple

```json
[
  {
    "house_id": 1,
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "median_house_value": 452600,
    "ocean_proximity": "NEAR BAY"
  }
]
```

### POST `/houses`

Add a new house entry to the `houses` table.

#### Request Body Exemple

```json
{
  "house_id": 2,
  "longitude": -122.22,
  "latitude": 37.86,
  "housing_median_age": 21,
  "total_rooms": 7099,
  "total_bedrooms": 1106,
  "population": 2401,
  "households": 1138,
  "median_income": 8.3014,
  "median_house_value": 358500,
  "ocean_proximity": "NEAR BAY"
}
```

#### Response

```json
{
  "message": "House added successfully!"
}
```

---

## Notes

- The application automatically checks for the existence of the `houses` table and the `house` database. If not found, they are created at runtime.
- Errors during runtime are logged to the console. Enable Flask's debug mode for detailed logs.

---

## Testing the API

### GET Example

```bash
curl http://127.0.0.1:5000/houses
```

### POST Example

```bash
curl -X POST http://127.0.0.1:5000/houses -H "Content-Type: application/json" -d '{
  "house_id": 3,
  "longitude": -122.24,
  "latitude": 37.85,
  "housing_median_age": 25,
  "total_rooms": 1467,
  "total_bedrooms": 190,
  "population": 496,
  "households": 177,
  "median_income": 7.2574,
  "median_house_value": 352100,
  "ocean_proximity": "INLAND"
}'
```

---

## Troubleshooting

### Common Issues

- **Connection Error**: Ensure PostgreSQL is running and credentials are correct.
- **Module Not Found**: Run `pip install -r requirements.txt` to install dependencies.

### Logs

All logs are printed to the console. Use `debug=True` in the `app.run()` method to see detailed error messages during development.

---

## License

This project is licensed under the MIT License.
