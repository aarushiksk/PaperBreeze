from agents import update_text, clean_text, paragraph_to_sentences, sentence_chunks_to_semantic_chunks, semantic_chunks_to_embeddings
from langgraph.graph import StateGraph, START,END
from typing_extensions import Optional, TypedDict
import numpy as np


class AgentState(TypedDict):
    text: Optional[str] = None
    chunks: Optional[list] = None
    embeddings: Optional[np.ndarray] = None
    query: Optional[str] = None
    response: Optional[str] = None
    
def create_graph():
    graph_builder = StateGraph(AgentState)
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



    return graph_builder.compile()
