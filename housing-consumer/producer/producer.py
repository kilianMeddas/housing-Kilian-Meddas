from confluent_kafka import Producer
import json
import time
import random

# Fonction pour créer un producteur Kafka
def create_producer():
    for _ in range(10):  # Essayer de se connecter pendant 10 tentatives
        try:
            producer = Producer({'bootstrap.servers': 'broker:9092'})
            print("Kafka Producer connecté avec succès.")
            return producer
        except Exception as e:
            print(f"Tentative de connexion échouée : {e}. Retente dans 5 secondes.")
            time.sleep(5)
    raise Exception("Impossible de se connecter au broker Kafka après 10 tentatives.")

# Initialisation du producteur Kafka
producer = create_producer()

# Fonction pour générer une maison avec des valeurs aléatoires
def generate_random_house():
    return {
        "house_id": random.randint(1, 100000),
        "longitude": round(random.uniform(-124.35, -114.31), 5),
        "latitude": round(random.uniform(32.54, 41.95), 5),
        "housing_median_age": random.randint(1, 52),
        "total_rooms": random.randint(500, 10000),
        "total_bedrooms": random.randint(100, 5000),
        "population": random.randint(200, 5000),
        "households": random.randint(100, 2000),
        "median_income": round(random.uniform(1.0, 15.0), 4),
        "median_house_value": random.randint(50000, 500000),
        "ocean_proximity": random.choice(["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]),
    }

# Fonction pour envoyer des messages au broker Kafka
def send_housing_data():
    while True:
        house_data = generate_random_house()
        
        # Envoi du message au topic Kafka
        producer.produce('housing_topic', value=json.dumps(house_data))
        producer.poll(0)  # Déclencher les callbacks de confirmation de livraison
        print(f"Message envoyé : {house_data}")

        time.sleep(2)  # Pause entre les envois

# Lancer l'envoi continu des messages
if __name__ == '__main__':
    send_housing_data()
