import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from vector_store import initialize_pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import time
import warnings
warnings.filterwarnings("ignore")

embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
index=initialize_pinecone()
print("index created")




def update_text(state):
    text=state["text"]
    cleaned_text = text.replace('\xa0', ' ')  
    return {'text': cleaned_text}


def clean_text(state):
    text=state['text']
    cleaned_text = text.replace('\xa0', ' ')  
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  
    cleaned_text = cleaned_text.strip()
    state['text']=cleaned_text
    print('\n\n\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    return {'text': state['text']}





def paragraph_to_sentences(state):
    text=state['text']
    sentences=re.split(r'(?<=[.])\s+', text)
    state['chunks']=sentences
    print("\n\n\n\n\n")
    print("<=------------------------------------------------------------------------------------------------------=>")
    print("Sentences:", len(sentences))
    
    return {'chunks': sentences}


    
def sentence_chunks_to_semantic_chunks(state):
    hf_embeddings = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(hf_embeddings)
    docs = text_splitter.create_documents([state['text']])
   
    text_contents = [doc.page_content for doc in docs]
    print('\n\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    print(text_contents)
    return {'chunks': text_contents}



def semantic_chunks_to_embeddings(state):
    documents=state['chunks']
    vectors_list = []  # List of document embeddings to upsert
    
    for i, document in enumerate(documents):
        embedding = embeddings.encode(document).tolist()  # Convert embedding to list for Pinecone
        doc_id = f"doc_{i}"  # Create a unique ID for each document

        # Upsert the vector into Pinecone with metadata
        vectors_list.append({
            "id": doc_id,
            "values": embedding,
            "metadata": {"text": document}  # Store the document's text as metadata
        })
    
    # Upsert vectors to Pinecone
    index.upsert(vectors=vectors_list)
    
    state['embeddings'] = vectors_list  # Store the embeddings in the state for future use
    return {'embeddings': vectors_list}