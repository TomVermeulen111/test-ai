from typing import List

from app.api.init_services import init_retriever
from langchain_core.documents import Document


def search_in_vector_store(question: str) -> List[Document]:
    retriever = init_retriever()
    return retriever.invoke(question)