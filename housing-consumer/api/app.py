from flask import Flask, request, jsonify
import psycopg2 as psycopg
import os
import time
import requests
app = Flask(__name__)

ML_MODEL_URL = "http://ml_service:5002/predict"  # Assuming Docker networking


# Récupération des variables d'environnement pour PostgreSQL
password = os.environ['POSTGRES_PASSWORD']
port = os.environ.get('DB_PORT', '5432')  # 5432 par défaut si non défini

# Fonction de création d'une connexion à PostgreSQL
def create_connection():
    for _ in range(10):  # Essayer de se connecter pendant 10 tentatives
        try:
            conn = psycopg.connect(database="postgres", user="postgres", password=password, host="db", port=port)
            return conn
        except Exception as e:
            print(f"Échec de connexion : {e}. Nouvelle tentative dans 5 secondes.")
            time.sleep(5)
    raise Exception("Impossible d'accéder à PostgreSQL après 10 tentatives.")

# Connexion initiale à la base 'postgres' pour vérifier/créer la base 'house'
conn = create_connection()
conn.autocommit = True
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", ('house',))
exists = cur.fetchone()
if not exists:
    cur.execute('CREATE DATABASE house;')
    print("Base de données 'house' créée avec succès !")
else:
    print("La base de données existe déjà.")
cur.close()
conn.close()

# Reconnexion à la base 'house'
conn = create_connection()
conn.autocommit = True
cur = conn.cursor()

# Fonction pour vérifier si une table existe
def table_exists(table_name):
    cur.execute('''
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    ''', (table_name,))
    return cur.fetchone()[0]

# Création de la table 'houses' si elle n'existe pas
if not table_exists('houses'):
    cur.execute('''
        CREATE TABLE houses (
            house_id INT UNIQUE,
            longitude FLOAT,
            latitude FLOAT,
            housing_median_age INT,
            total_rooms INT,
            total_bedrooms INT,
            population INT,
            households INT,
            median_income FLOAT,
            median_house_value FLOAT, 
            ocean_proximity VARCHAR(255),
            PRIMARY KEY(house_id)
        );
    ''')
    print("Table 'houses' créée avec succès !")
else:
    print("La table 'houses' existe déjà.")

# Fonction pour récupérer toutes les maisons
def get_all_houses():
    cur.execute('SELECT * FROM houses')
    houses = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    house_list = []
    for house in houses:
        house_list.append(dict(zip(columns, house)))
    return house_list

# Route GET pour récupérer toutes les maisons
@app.route('/houses', methods=['GET'])
def get_houses():
    houses = get_all_houses()
    return jsonify(houses)

# Route POST pour ajouter une maison
@app.route('/houses', methods=['POST'])
def set_house():
    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "Aucune donnée JSON fournie"}), 400

    # Si les données sont enveloppées dans une clé "data", on les extrait
    if "data" in request_data:
        request_data = request_data["data"]

    query = '''
        INSERT INTO houses (
            house_id, longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
            population, households, median_income, median_house_value, ocean_proximity
        ) 
        VALUES (
            %(house_id)s, %(longitude)s, %(latitude)s, %(housing_median_age)s, %(total_rooms)s, %(total_bedrooms)s, 
            %(population)s, %(households)s, %(median_income)s, %(median_house_value)s, %(ocean_proximity)s
        )
        ON CONFLICT (house_id) DO NOTHING;
    '''
    try:
        cur.execute(query, request_data)
        return jsonify({"message": "Maison ajoutée avec succès !"}), 201
    except Exception as e:
        print(f"Erreur lors de l'insertion : {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    data = request.get_json()
    response = requests.post(ML_MODEL_URL, json=data)
    
    return jsonify(response.json()), response.status_code


if __name__ == '__main__':
    print('''
Pour vérifier la connexion :
  - GET : curl http://localhost:5000/houses
  - POST : curl -X POST http://localhost:5000/houses -H "Content-Type: application/json" -d '{"data": {"house_id": 1, "longitude": 1.23, "latitude": 2.34, "housing_median_age": 20, "total_rooms": 5, "total_bedrooms": 3, "population": 10, "households": 2, "median_income": 3.5, "median_house_value": 250000, "ocean_proximity": "NEAR BAY"}}'
    ''')
    app.run(host='0.0.0.0', port=5000, debug=True)