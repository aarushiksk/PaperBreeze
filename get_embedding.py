import requests
import os
import time
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"

# Function to wait until HF_TOKEN is available
def wait_for_token():
    HF_TOKEN = os.environ.get('HF_TOKEN')
    while not HF_TOKEN:
        print("Waiting for HF_TOKEN to be set...")
        time.sleep(5)  
        load_dotenv() 
        HF_TOKEN = os.environ.get('HF_TOKEN')
    print("HF_TOKEN obtained!")
    return HF_TOKEN

hf_token = wait_for_token()

headers = {"Authorization": "Bearer " + hf_token}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def enter_to_embed(text):
    output = query({
        "inputs": text,
    })
    return output


