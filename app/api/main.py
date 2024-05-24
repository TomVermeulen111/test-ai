from fastapi import FastAPI
from dotenv import load_dotenv
from search_in_vector_store import search_in_vector_store

app = FastAPI()

load_dotenv()

@app.get("/search_in_vector_store/{question}")
def get_search_in_vector_store(question: str):
    return search_in_vector_store(question)


@app.get("/retrieval_augmented_generation")
def get_retrieval_augmented_generation():
    return {"item_id": "q"}         