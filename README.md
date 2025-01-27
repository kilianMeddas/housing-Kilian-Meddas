# Flask and PostgreSQL Integration with Python

## Overview

This project demonstrates how to integrate a Python Flask web application with a PostgreSQL database. It includes tools for database migrations and creation using Flask-Migrate and psycopg2.

---

## Architecture

The project is organized as follows:

```
housing-Kilian-Meddas/
|
|-- housing-model/
|   |-- Dockerfile
|   |-- api.py
|   |-- housing.csv
|   |-- requirements.txt
|
|-- housing_api/
|   |-- venv/
|   |-- app.py
|   |-- app_migration.py
|   |-- create_db.py
|   |-- requirements.txt
|
|-- housing_api_docker/
|   |-- api/
|       |-- Dockerfile
|       |-- app_migration.py
|       |-- requirements.txt
|
|-- create_database/
|   |-- Dockerfile
|   |-- create_db.py
|   |-- requirements.txt
|
|-- docker-compose.yml
|-- instru_docker
|-- .gitignore
|-- LICENSE
|-- README.md
```
---

## Prerequisites

### Tools and Libraries

- Docker
- Docker Compose
- PostgreSQL
- Python
---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kilianMeddas/housing-kilian-meddas-docker.git
cd housing-kilian-meddas-docker
cd housing_api
```

### Run Docker Compose (Only for housing_api_docker)

Ensure Docker and Docker Compose are installed on your machine. Then, run:

```bash
docker-compose up --build
```

This will build and start the services defined in the `docker-compose.yml` file, including the Flask application and PostgreSQL database.

---

**For next things, you need to execute `docker exec -it house_api bash`**

## Create the table 

Execute : 
` flask db init`
` flask db migrate -m "Create houses table"`
` flask db upgrade`
  `

## API Endpoints

### GET `/houses`

Retrieve all house entries from the `houses` table.

#### Response Example

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

#### Request Body Example

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

---------------------------
---------------------------
---------------------------

### Docker for housing-model

```docker build -t docker-ml-model -f Dockerfile .```

``` docker run -v $(pwd)/output:/output docker-ml-model```
## Troubleshooting

### Common Issues

- **Docker Error**: Ensure Docker and Docker Compose are installed and running.
- **Connection Error**: Ensure PostgreSQL is correctly configured in the Docker Compose file.

### Logs

All logs are printed to the console. Use Docker logs to inspect issues:

```bash
docker-compose logs
```

---

## License

This project is licensed under the MIT License.

------------------------

