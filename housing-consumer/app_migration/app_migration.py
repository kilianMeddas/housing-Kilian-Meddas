from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
import time
password = os.environ['POSTGRES_PASSWORD']
port = os.environ.get('DB_PORT', '5432')  # 5432 if not set
Flask_port = "5000"
SQLAlchemy_URI = os.environ.get('SQLALCHEMY_DATABASE_URI', f'postgresql://postgres:{password}@db:{port}/house')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLAlchemy_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy()
migrate = Migrate(app, db)


# Fonction pour tester la connexion à la base de données
def wait_for_db():
    for attempt in range(10):  # Maximum 10 tentatives
        try:
            engine = db.engine  # Crée un moteur pour tester la connexion
            with engine.connect() as connection:
                print("Connexion réussie à PostgreSQL.")
                return
        except Exception as e:
            print(f"Tentative {attempt + 1}/10 : Échec de connexion - {e}")
            time.sleep(5)
    raise Exception("Impossible de se connecter à PostgreSQL après 10 tentatives.")


# Initialisation après configuration
db.init_app(app)
with app.app_context():
    wait_for_db()


class House(db.Model):
    __tablename__ = "houses"
    house_id = db.Column(db.Integer, primary_key=True, unique=True)
    longitude = db.Column(db.Float)
    latitude = db.Column(db.Float)
    housing_median_age = db.Column(db.Integer)
    total_rooms = db.Column(db.Integer)
    total_bedrooms = db.Column(db.Integer)
    population = db.Column(db.Integer)
    households = db.Column(db.Integer)
    median_income = db.Column(db.Float)
    median_house_value = db.Column(db.Float)
    ocean_proximity = db.Column(db.String(255))


# Routes
@app.route('/houses', methods=['GET'])
def get_houses():
    houses = House.query.all()
    return jsonify([{
        "house_id": house.house_id,
        "longitude": house.longitude,
        "latitude": house.latitude,
        "housing_median_age": house.housing_median_age,
        "total_rooms": house.total_rooms,
        "total_bedrooms": house.total_bedrooms,
        "population": house.population,
        "households": house.households,
        "median_income": house.median_income,
        "median_house_value": house.median_house_value,
        "ocean_proximity": house.ocean_proximity,
    } for house in houses])


@app.route('/houses', methods=['POST'])
def add_house():
    data = request.json
    house = House(**data)
    db.session.add(house)
    db.session.commit()
    return jsonify({"message": "House added successfully!"}), 201


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Flask_port, debug=True)
