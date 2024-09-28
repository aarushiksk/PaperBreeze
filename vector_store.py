from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time
index = None
pc=None
index_name=None
def initialize_pinecone():
    global index,pc, index_name
    load_dotenv()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    environment=os.environ.get("PINECONE_ENVIRONMENT")
    pc = Pinecone(api_key=pinecone_api_key, environment=environment)
    
    index_name = "langchain-test-index" 
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
      pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(index_name)
    
    return index

def get_index():
    if index is None:
        raise ValueError("Pinecone index has not been initialized. Please call initialize_pinecone() first.")
    
    return index

def Delete_Index():
    pc.delete_index(index_name)

    return "Your Document has been deleted! Thank you for using PaperBreeze📃🍃"