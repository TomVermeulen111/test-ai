import sys
from typing import Literal
sys.path.insert(0, "app")
sys.path.insert(0, "app/api")
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.search import search
from app.api.retrieval_augmented_generation import retrieval_augmented_generation

app = FastAPI()

load_dotenv()

default_system_prompt="""You are an assistant for question-answering tasks. 
                                     
You can use the following pieces of retrieved context to answer the question. 
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in dutch."""

@app.get("/search/{question}")
def get_search(question: str, 
    schema_filter: str | None = None,
    order_by_date: bool = False,
    addVectors: bool = False, 
    search_type: Literal['hybrid_search','similarity_search', 'vector_search', 'simple_text']='similarity_search', 
    nr_of_documents_to_return=10,
    ):
    return search(question, addVectors, search_type, nr_of_documents_to_return, schema_filter, order_by_date)


@app.get("/retrieval_augmented_generation")
def get_retrieval_augmented_generation(
    question: str, 
    top_k: int = 3, 
    score_threshold: float = 0.7, 
    system_prompt: str=default_system_prompt, 
    context: str="CIB_MEMBER",
    include_page_content: bool = False
):
    return retrieval_augmented_generation(question, top_k, score_threshold, system_prompt, context, include_page_content)