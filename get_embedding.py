import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
headers = {"Authorization": "Bearer"+" "+os.environ.get('HF_TOKEN')}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
def enter_to_embed(str):
 output = query({
	"inputs": "hi",
 })
 return output

 
