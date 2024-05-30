from typing import List

from app.api.init_services import init_retriever
from langchain_core.documents import Document


def search_in_vector_store(question: str, addVectors: bool) -> List[Document]:
    retriever = init_retriever(k=3)
    documents = retriever.invoke(question)
    if not addVectors:
        for d in documents:
            d.metadata['content_vector']=[]
    return documents