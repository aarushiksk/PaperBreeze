from fastapi import FastAPI, UploadFile, File,Request,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,RedirectResponse
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict, Optional
from groq import Groq
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from PyPDF2 import PdfReader
from fastapi.templating import Jinja2Templates
import numpy as np




import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()   

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


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
# print(index.describe_index_stats())




embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")



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
    text=state["text"]
    return {'text': text}

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
#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=100
# )

#     text=state['text']
#     print('\n\n\n')
#     print("<=------------------------------------------------------------------------------------------------------=>")
    
#     print("Length",len(text))
    
#     data = text_splitter.create_documents([text])
#     print(data)
    
#     doc =[]
#     for d in data:
#         d = d.page_content
#         doc.append(d)


    
#     print("docs",len(doc))
    
#     print("Docs",doc)
    
#     print('\n\n\n')
#     print("<=------------------------------------------------------------------------------------------------------=>")
    hf_embeddings = HuggingFaceEmbeddings()
    text_splitter = SemanticChunker(hf_embeddings)
    docs = text_splitter.create_documents([state['text']])
   
    text_contents = [doc.page_content for doc in docs]
    print('\n\n\n\n')
    print("<=------------------------------------------------------------------------------------------------------=>")
    print(text_contents)
    return {'chunks': text_contents}

def semantic_chunks_to_embeddings(state: State):
    documents=state['chunks']
    vectors_list = []  # List of document embeddings to upsert
    metadata_list = []  # List to store document metadata (ID and content)
    
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




class State2(TypedDict):
    query: Optional[str] = None
    response: Optional[str] = None
    
graph_builder2 = StateGraph(State2)
    
def get_response(state:State2):
    query_text = state['query']
    # Convert the query text to embeddings
    query_embedding = embeddings.encode(query_text).tolist()
    
    # Query Pinecone for similar embeddings
    results = index.query(
        vector=query_embedding,
        top_k=1,  # Number of results to return
        include_metadata=True  # Retrieve the associated metadata (e.g., text)
    )
    
    # Extract the top responses (e.g., document text)
    response_texts = [result['metadata']['text'] for result in results['matches']]
    
    # Set the response in the state
    state['response'] = response_texts
    chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        {
            "role": "system",
            "content": "you are a helpful research paper assistant, who simplifies things for user"
        },
        {
            "role": "user",
            "content":query_text+"  "+"response from external source"+" "+response_texts[0],
        }
    ],
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=1024,
    top_p=1,
    stop=None,
    stream=False,
)

# Print the completion returned by the LLM.
    resp=chat_completion.choices[0].message.content
    return {'response': resp}

graph_builder2.add_node("get_response", get_response)
graph_builder2.add_edge(START, "get_response")
graph_builder2.add_edge("get_response", END)
graph_app2 = graph_builder2.compile()

    
@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index (1).html", {"request": request})


 

@app.post("/upload", response_class=HTMLResponse)
def upload(request: Request, file: UploadFile = File(...)):
    # Extract text from the uploaded PDF
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Initial state with the extracted text
    initial_state = {"text": text, "chunks": None, "embeddings": None, "query": None, "response": None}
    
    # Process the text using the graph
    result = graph_app.invoke(initial_state)
    
    # Render the chatbot page after processing
    return RedirectResponse(url='/chatbot')


@app.post("/chatbot", response_class=HTMLResponse)
def get_chatbot(request: Request):
    
    # This will render the chatbot page without any data initially
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.post("/answer", response_class=HTMLResponse)
def handle_query(request: Request, query: str = Form(...)):
    # Handle the user's query by invoking the second state graph
    state = {"query": query, "response": None}
    result = graph_app2.invoke(state)
    
    # Return the chatbot template with the response data
    return templates.TemplateResponse("chatbot.html", {"request": request, "query":query,"response": result['response']})

   

