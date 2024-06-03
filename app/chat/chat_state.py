from typing import List
from langchain_core.documents import Document

class ChatState:
    chain_id: str = ""
    question: str = ""
    answer: str = ""
    prompt: str = ""
    documents: List[Document] = []