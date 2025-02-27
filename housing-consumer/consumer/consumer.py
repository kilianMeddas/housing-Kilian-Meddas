from confluent_kafka import Consumer, KafkaError
import json
import requests
import time

# Fonction pour créer un consumer Kafka
def create_consumer():
    conf = {
        'bootstrap.servers': 'broker:9092',
        'group.id': 'house_api_group',
        'auto.offset.reset': 'earliest',       # Démarrage en début de topic si aucun offset connu
        'enable.auto.commit': True,            # Commit automatique des offsets
    }
    for _ in range(10):  # Essayer de se connecter pendant 10 tentatives
        try:
            consumer = Consumer(conf)
            # S'abonner au topic souhaité
            consumer.subscribe(['housing_topic'])
            print("Kafka Consumer connecté avec succès.")
            return consumer
        except Exception as e:
            print(f"Tentative de connexion échouée : {e}. Retente dans 5 secondes.")
            time.sleep(5)
    raise Exception("Impossible de se connecter au broker Kafka après 10 tentatives.")

api = "http://house_api:5000/houses"
consumer = create_consumer()

# Fonction de consommation des messages
def consume_messages():
    print("Démarrage de la consommation Kafka...")
    try:
        while True:
            msg = consumer.poll(1.0)  # Attente d'un message pendant 1 seconde
            if msg is None:
                continue  # Aucun message reçu, on recommence
            if msg.error():
                # Gestion d'erreur éventuelle
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue  # Fin de partition, pas de problème
                else:
                    print(f"Erreur lors de la consommation : {msg.error()}")
                    continue

            # Traitement du message reçu
            try:
                # Le contenu du message est en bytes, on le décode et on essaie de le parser en JSON
                message_str = msg.value().decode('utf-8')
                message_data = json.loads(message_str)
            except Exception as parse_error:
                print(f"Erreur lors du parsing du message : {parse_error}")
                message_data = msg.value().decode('utf-8')

            # Envoi du message à l'API via une requête POST
            try:
                response = requests.post(api, json={'data': message_data})
                response.raise_for_status()
                print(f"Message envoyé à l'API : {message_data}, Response: {response.status_code}")
            except requests.exceptions.RequestException as req_err:
                print(f"Échec de l'envoi du message : {req_err}")

    except KeyboardInterrupt:
        print("Consommation interrompue par l'utilisateur.")
    finally:
        consumer.close()

if __name__ == '__main__':
    consume_messages()
