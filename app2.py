from fastapi import FastAPI, Form, UploadFile, File, JSONResponse
from fastapi.responses import HTMLResponse
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_groq import ChatGroq
import os
from langgraph.checkpoint.memory import MemorySaver
from PyPDF2 import PdfReader


memory = MemorySaver()
app2 = FastAPI()









# Define the chatbot state using LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    text:str
    embeddings: str
    query: str
    response:str


graph_builder = StateGraph(State)
llm = ChatGroq(
    temperature=0,
    model_name="Llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Define the chatbot node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

@app2.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Process the file content as needed
    return text
  
# Serve the HTML form
@app2.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot</title>
      </head>
      <body>
        <h1>Chat with LangGraph Bot</h1>
        <form action="/chat" method="post">
          <label for="user_input">Your question:</label>
          <input type="text" id="user_input" name="user_input" required>
          <button type="submit">Send</button>
        </form>
      </body>
    </html>
    """

# Process the input from the form
@app2.post("/chat")
async def chat(user_input: str = Form(...)):
    # Send user input to the LangGraph chatbot
    
    config = {"configurable": {"thread_id": "1"}}
    # Process through LangGraph
    response_message = ""
    events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
    for event in events:
            response_message = event["messages"][-1].content  
    # # Get last message from assistant
    
    # response_message=llm.invoke(state["messages"])

    # Return the chatbot's response as an HTML response
    return HTMLResponse(f"<h1>Assistant:</h1><p>{response_message}</p><br><a href='/'>Ask another question</a>")
