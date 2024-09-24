from fastapi import FastAPI, UploadFile, File,Request
from fastapi.responses import HTMLResponse
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from PyPDF2 import PdfReader
from fastapi.templating import Jinja2Templates
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tokenizers import Tokenizer



import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


app = FastAPI()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
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
# print(index.describe_index_stats())




load_dotenv()   


embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")



templates = Jinja2Templates(directory="templates/")

class State(TypedDict):
    
    text: Optional[str] = None
    chunks: Optional[list] = None
    embeddings: Optional[np.ndarray] = None
    query: Optional[str] = None
    response: Optional[str] = None

graph_builder = StateGraph(State)

llm = ChatGroq(
    temperature=0,
    model_name="Llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
)

def update_text(state: State):
    state["text"]=state['text']
    return {'text': state['text']}

def clean_text(state: State):
    text=state['text']
    cleaned_text = text.replace('\xa0', ' ')  # Replace non-breaking space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces/newlines/tabs with a single space
    cleaned_text = cleaned_text.strip()
    state['text']=cleaned_text
    return {'text': state['text']}


def paragraph_to_sentences(state: State):
    text=state['text']
    sentences=re.split(r'(?<=[.])\s+', text)
    state['chunks']=sentences
    print("\n\n\n\n\n")
    print("<=------------------------------------------------------------------------------------------------------=>")
    print("Sentences:", len(sentences))
    
    return {'chunks': sentences}
    
def sentence_chunks_to_semantic_chunks(state: State):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0
)

    text=state['text']
    print('\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    
    print("Length",len(text))
    
    data = text_splitter.create_documents([text])
    print(data)
    
    doc =[]
    for d in data:
        d = d.page_content
        doc.append(d)


    
    print("docs",len(doc))
    
    print("Docs",doc)
    
    print('\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    

    return {'chunks': len(data)}

def semantic_chunks_to_embeddings(state: State):
    documents=state['chunks']
    vectors_list = []  # List of document embeddings to upsert
    j=0
    for i in len(documents):
        embedding=embeddings.encode(documents[i]).tolist()[0]
        print(embedding)
        vectors_list.append({"id": str(j), "values": embedding})
        j+=1
    print('\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    print(len(vectors_list))
    index.upsert(vectors=vectors_list)
    return {'embeddings': vectors_list}


graph_builder.add_node("update_text",update_text )
graph_builder.add_node("clean_text", clean_text)
graph_builder.add_node("paragraph_to_sentences", paragraph_to_sentences)
graph_builder.add_node("sentence_chunks_to_semantic_chunks", sentence_chunks_to_semantic_chunks)
graph_builder.add_node("semantic_chunks_to_embeddings", semantic_chunks_to_embeddings)
graph_builder.add_edge(START, "update_text")
graph_builder.add_edge("update_text", "clean_text")
graph_builder.add_edge("clean_text", "paragraph_to_sentences")
graph_builder.add_edge("paragraph_to_sentences", "sentence_chunks_to_semantic_chunks")
graph_builder.add_edge("sentence_chunks_to_semantic_chunks", "semantic_chunks_to_embeddings")
graph_builder.add_edge("semantic_chunks_to_embeddings", END)


graph_app = graph_builder.compile()
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    # Extract text from the uploaded PDF
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Initial state with the extracted text
    initial_state = {"text": text, "chunks": None, "embeddings": None, "query": None, "response": None}
    
    # Invoke the StateGraph with the initial state

    
    # Display a message after all operations are performed
    result = graph_app.invoke(initial_state)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

