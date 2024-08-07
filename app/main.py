import sys
sys.path.append('.')
from typing import List, Literal
from fastapi import FastAPI, Response
from dotenv import load_dotenv
from api.search import search
from api.retrieval_augmented_generation import retrieval_augmented_generation
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel
from api.chat import chat    
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

app = FastAPI()

load_dotenv()

class ChatMessage(BaseModel):
    role: Literal['system', 'human']
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
    context:List[dict]

default_system_prompt="""Act as a professional assistant for CIB that answers questions of our members who are mainly realestate brokers and syndics.
Your instructions are to help the CIB-members with all their questions, from general questions, questions about CIB organization, online tools, juridical questions etc.
The end goal is that the conversation partner is well informed and doesn't need to ask the question to a human (legal) expert in real-estate.
You can only use the following pieces of retrieved context to answer the question.
If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question.
If you find multiple answers or if your answer would be too generic, ask the user to specify his question more. Indicate where he needs to specify.
Use four sentences maximum and keep the answer concise and don't use overly flawed language.
You will have a chat history, but you must only answer the last question.
You MUST answer in dutch.
The date of today is: """ + str(datetime.now()), 

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

@app.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest):
    history = InMemoryChatMessageHistory()
    for message in request.chat_history:
        if message.role == 'system':
            history.add_message(SystemMessage(message.content))
        else:
            history.add_message(HumanMessage(message.content))
    
    result = chat(request.question, history)
    print(result['context'])
    return ChatResponse(answer=result['answer'], context=[d.metadata for d in result['context']])