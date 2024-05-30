import sys
sys.path.insert(0, "app")
sys.path.insert(0, "app/api")
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.search_in_vector_store import search_in_vector_store
from app.api.retrieval_augmented_generation import retrieval_augmented_generation

app = FastAPI()

load_dotenv()

default_system_prompt="""You are an assistant for question-answering tasks. 
                                     
You can use the following pieces of retrieved context to answer the question. 
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in dutch."""

@app.get("/search_in_vector_store/{question}")
def get_search_in_vector_store(question: str, addVectors: bool = False):
    return search_in_vector_store(question, addVectors)


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