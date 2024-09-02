import sys

from pygments import highlight
sys.path.append('.')
from api.filters import SearchFilters
from typing import List, Literal
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from api.search import SearchResult, search
from api.retrieval_augmented_generation import retrieval_augmented_generation
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel
from api.chat import chat    
from langchain_core.messages import HumanMessage, SystemMessage
from api.wegov_integration import validate_partner_key

header_scheme = APIKeyHeader(name="Authorization")

# Dependency to validate the partner key
def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(header_scheme)):
    if not validate_partner_key(credentials):
        raise HTTPException(status_code=403, detail="Invalid partner key")
    return credentials


app = FastAPI(dependencies=[Depends(get_api_key)])

load_dotenv()

class ChatMessage(BaseModel):
    role: Literal['system', 'human']
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
    context: List[dict]

class SearchRequest(BaseModel):
    question: str
    order_by_date: bool = False
    addVectors: bool = False
    search_type: Literal['hybrid_search', 'similarity_search', 'vector_search', 'simple_text'] = 'similarity_search'
    filters: SearchFilters = None
    top: int = 10
    skip: int = 0

class SearchDocument(BaseModel):
    page_content: str
    metadata: dict
    highlights: dict
    score: float

class SearchResponse(BaseModel):
    results: List[SearchDocument]
    total: int


default_system_prompt = """You are an assistant for question-answering tasks. 
                                     
You can use the following pieces of retrieved context to answer the question. 
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in Dutch."""

@app.post("/api/search", response_model=SearchResponse)
def post_search(request: SearchRequest ):
    searchResult = search(request.question, request.addVectors, request.search_type, request.order_by_date, request.filters, request.top, request.skip)
    response = SearchResponse(total=searchResult.count, results=[])
    for d in searchResult.results:
        response.results.append(SearchDocument(page_content=d.document.page_content, metadata=d.document.metadata or dict(), highlights=d.highlights or dict(), score=d.score))
    return response

@app.get("/api/retrieval_augmented_generation")
def get_retrieval_augmented_generation(
    question: str, 
    top_k: int = 3, 
    score_threshold: float = 0.7, 
    system_prompt: str = default_system_prompt, 
    context: str = "CIB_MEMBER",
    include_page_content: bool = False,
):
    return retrieval_augmented_generation(question, top_k, score_threshold, system_prompt, context, include_page_content)

@app.post("/api/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest):
    history = InMemoryChatMessageHistory()
    for message in request.chat_history:
        if message.role == 'system':
            history.add_message(SystemMessage(message.content))
        else:
            history.add_message(HumanMessage(message.content))
    
    result = chat(request.question, history)
    return ChatResponse(answer=result['answer'], context=[d.metadata for d in result['context']])
