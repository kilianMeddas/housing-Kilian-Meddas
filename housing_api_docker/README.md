# Housing API with Docker and PostgreSQL

This project provides a Flask API for managing housing data using PostgreSQL. It is fully containerized with Docker and orchestrated using Docker Compose.

## Project Structure

```
/housing_api
│-- /api
│   │-- Dockerfile
│   │-- app_migration.py
│   │-- requirements.txt
│
│-- /create_database
│   │-- Dockerfile
│   │-- create_db.py
│   │-- requirements.txt
│
│-- docker-compose.yml
```

### 1. API Service (`/api`)
- **Dockerfile**: Defines the API container.
- **app_migration.py**: Flask app using SQLAlchemy and Flask-Migrate.
- **requirements.txt**: Lists dependencies.

### 2. Database Initialization (`/create_database`)
- **Dockerfile**: Defines the database initialization container.
- **create_db.py**: Ensures the PostgreSQL database and tables exist.
- **requirements.txt**: Lists dependencies.

### 3. Docker Compose (`docker-compose.yml`)
Manages services:
- `db`: PostgreSQL database.
- `house_api`: Flask API service.
- `create_db`: Service to initialize the database before the API starts.

## Setup and Deployment

### Build and Push Docker Images
```sh
docker build -t kmeddas/house_api ./api
docker build -t kmeddas/create_database ./create_database
docker push kmeddas/house_api
docker push kmeddas/create_database
```

### Cleanup Before Deployment (Optional)
```sh
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q) --force
docker volume rm $(docker volume ls -q)
docker network rm $(docker network ls -q)
```

### Start Services with Docker Compose
```sh
docker compose up
```

## Environment Variables
Set in `docker-compose.yml`:
- `POSTGRES_PASSWORD`: Database password.
- `DB_PORT`: PostgreSQL port.
- `SQLALCHEMY_DATABASE_URI`: Database connection URI for Flask.

## Access the API
After deployment, access the API at:
```
http://localhost:5000/houses
```

