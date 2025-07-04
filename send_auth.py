from dotenv import load_dotenv
import os


def get_api_key():
    load_dotenv() 
    api_key = os.getenv("Authoruzation_key") 
    return api_key
    