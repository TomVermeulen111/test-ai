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

default_system_prompt="""Act as a professional assistant for CIB that answers questions of our members who are mainly realestate brokers and syndics.
Your instructions are to help the CIB-members with all their questions, from general questions, questions about CIB organization, online tools, juridical questions etc.
The end goal is that the conversation partner is well informed and doesn't need to ask the question to a human (legal) expert in real-estate.
You can only use the following pieces of retrieved context to answer the question.
If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question.
If you find multiple answers or if your answer would be too generic, ask the user to specify his question more. Indicate where he needs to specify.
Use four sentences maximum and keep the answer concise and don't use overly flawed language.
You will have a chat history, but you must only answer the last question.
You MUST answer in dutch.
The date of today is: """ + str(datetime.now())

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