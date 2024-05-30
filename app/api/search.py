from typing import Dict, List, Literal, Tuple

from app.api.init_services import init_search_client, init_vector_store
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
import json
from azure.search.documents import SearchItemPaged

# Allow overriding field names for Azure Search
FIELDS_ID = get_from_env(
    key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
)
FIELDS_CONTENT = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT",
    env_key="AZURESEARCH_FIELDS_CONTENT",
    default="content",
)
FIELDS_CONTENT_VECTOR = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    default="content_vector",
)
FIELDS_METADATA = get_from_env(
    key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
)


def search(search_query: str, type: Literal['hybrid','semantic','text'], k) -> List[Document]:
    client = init_search_client()

    vector_store = init_vector_store()

    if type == 'hybrid':
        documents = vector_store.hybrid_search(search_query, k)
    elif type == 'semantic':
        documents = vector_store.similarity_search(search_query, k)
    elif type == 'text':
        documents = client.search(search_text=search_query, top=k)
    
    vector_store.semantic_hybrid_search

    return documents

def _results_to_documents(
    results: SearchItemPaged[Dict],
) -> List[Tuple[Document, float]]:
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
        )
        for result in results
    ]
    return docs

def _result_to_document(result: Dict) -> Document:
    return Document(
        page_content=result.pop(FIELDS_CONTENT),
        metadata=json.loads(result[FIELDS_METADATA])
        if FIELDS_METADATA in result
        else {
            key: value for key, value in result.items() if key != FIELDS_CONTENT_VECTOR
        },
    )
