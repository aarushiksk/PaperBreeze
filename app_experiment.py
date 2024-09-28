from graph import create_graph
from langgraph.graph import StateGraph, START,END
from typing_extensions import Optional, TypedDict
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File,Request,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,RedirectResponse
from typing import Literal
from vector_store import get_index,Delete_Index
from groq import Groq
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from fastapi.templating import Jinja2Templates
from guardrails.hub import ProfanityFree, ToxicLanguage
from guardrails import Guard
import os


app_experiment = FastAPI()
app_experiment.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

load_dotenv()
embeddings = SentenceTransformer("BAAI/bge-large-en-v1.5")
index=get_index()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
os.environ['GEMINI_API_KEY'] = os.environ.get("GEMINI_API_KEY")




class UserInput(TypedDict):
    query: Optional[str] = None
    response: Optional[str] = None  
    
graph_builder2 = StateGraph(UserInput)


def input_query(state: UserInput):
    state['query'] = state['query']
    return {'query': state['query']}

def check(state: UserInput) -> Literal["guard_message", "get_response"]:
    query_text=state['query']
    guard = Guard().use_many(ToxicLanguage(threshold=0.5, validation_method="sentence"),  
                             ProfanityFree())

    validation=guard.validate(query_text)
    print(validation)
     
    validation_passed=validation.validation_passed
     
    if validation_passed:
         return "get_response"
    else:
        return "guard_message"
    
 
def guard_message(state: UserInput):
    state['response'] = "Sorry i can't answer that right now, please rephrase your prompt"   
    
    return {'response': state['response']}

def get_response(state: UserInput):
    query_text = state['query']
    query_embedding = embeddings.encode(query_text).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=1,  
        include_metadata=True  
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
            "content": '''You are a helpful research paper assistant, who simplifies things for user.You must understand the user and precisely answer only what has been asked, no extra information is to be included by you.
             If user greets, just greet back even if you are given response from external source.
             If user asks a question, answer it.
            '''
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


    resp=chat_completion.choices[0].message.content
    return {'response': resp}


def create_conversation_graph():
     graph_builder2.add_node("input_query", input_query)
     graph_builder2.add_node("guard_message", guard_message)
     graph_builder2.add_node("get_response", get_response)
     graph_builder2.add_edge(START, "input_query")
     graph_builder2.add_conditional_edges(
     "input_query",
      check,
      {"guard_message": "guard_message",
       "get_response": "get_response",
       },
)
     return graph_builder2.compile()


graph_app2=create_conversation_graph()


@app_experiment.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index (1).html", {"request": request})




@app_experiment.post("/upload", response_class=HTMLResponse)
def upload(request: Request, file: UploadFile = File(...)):
    # Extract text from the uploaded PDF
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    initial_state = {"text": text, "chunks": None, "embeddings": None, "query": None, "response": None}
    
    # Process the text using the graph
    graph_app=create_graph()
    result = graph_app.invoke(initial_state)
    
    # Render the chatbot page after processing
    return RedirectResponse(url='/chatbot')


@app_experiment.post("/chatbot", response_class=HTMLResponse)
def get_chatbot(request: Request):
    
    # This will render the chatbot page without any data initially
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app_experiment.post("/answer", response_class=HTMLResponse)
def handle_query(request: Request, query: str = Form(...)):
    # Handle the user's query by invoking the second state graph
    state = {"query": query, "response": None}
    result = graph_app2.invoke(state)
    
    # Return the chatbot template with the response data
    return templates.TemplateResponse("chatbot.html", {"request": request, "query":query,"response": result['response']})


@app_experiment.post("/end",response_class=HTMLResponse)
def handle_end(request: Request, end: str = Form(...)):
    response=Delete_Index()
    
    return templates.TemplateResponse("end.html", {"request": request, "response": response})
    

   
