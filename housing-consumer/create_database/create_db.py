import psycopg2 as psycopg
import os
import time

password = os.environ['POSTGRES_PASSWORD']
port = os.environ.get('DB_PORT', '5432')  # 5432 if not set


# Connect to the 'postgres' database (this is the default database in PostgreSQL)
def create_connexion():
    for _ in range(10):  # Essayer de se connecter pendant 10 tentatives
        try:
            conn = psycopg.connect(database="postgres", user="postgres", password=password, host="db", port=port)
            return conn
        except Exception as e:
            print(f"Connection attempt failure: {e}. Retry in 5 seconds.")
            time.sleep(5)
    raise Exception("Unable to access postgres after 10 attempts.")


# To create database "house" if not exists
conn = create_connexion()

# To ensure that every command will be executed
conn.autocommit = True
cur = conn.cursor()
# Check if the 'house' database exists, if not, create it
cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", ('house',))

exists = cur.fetchone() # If none : don't exists and create the database
if not exists:
    cur.execute('CREATE DATABASE house;')
    print("Database 'house' created successfully!")

elif exists:
    print("The database already exists")

# After checking the database, we go in
cur.close()
conn.close()