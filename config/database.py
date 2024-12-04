from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

db_read_host = os.getenv('DB_READ_HOST')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_port = os.getenv('DB_PORT')

DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_read_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_connection(): 
    return engine.connect()

def close_db_connection(connection): 
    connection.close()
