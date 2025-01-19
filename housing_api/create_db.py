import psycopg2 as psycopg


# Connect to the 'postgres' database (this is the default database in PostgreSQL)
# To create database "house" if not exists
conn = psycopg.connect(database="postgres", user="postgres", password="2005", host="127.0.0.1", port="5432")

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