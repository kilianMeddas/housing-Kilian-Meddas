# Housing API

## Project Structure
```
/housing_api
│-- app.py              # Main Flask application
│-- app_migration.py    # Flask app with SQLAlchemy and Flask-Migrate
│-- create_db.py        # Script to initialize the database
│-- requirements.txt    # Dependencies
```

## Installation
### Prerequisites
- Python 3
- PostgreSQL
- Virtual environment (recommended)

### Setup Instructions
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd housing_api
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Database Setup
1. Ensure PostgreSQL is running and create the database by executing:
   ```sh
   python create_db.py
   ```

2. If using Flask-Migrate, initialize migrations:
   ```sh
   flask db init
   flask db migrate -m "Initial migration."
   flask db upgrade
   ```

## Running the API
To start the API, run:
```sh
python app.py
```
The server will run at `http://localhost:5000/`.

## API Endpoints
### Retrieve all houses
**GET** `/houses`
- Returns a list of all houses in the database.

### Add a new house
**POST** `/houses`
- Expects JSON payload:
  ```json
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
  ```
- Response:
  ```json
  { "message": "House added successfully!" }
  ```

## Notes
- `app.py` manually handles database connections and queries.
- `app_migration.py` uses SQLAlchemy and Flask-Migrate for database management.
- `create_db.py` ensures the database exists before running the API.

## Author
Meddas Kilian

