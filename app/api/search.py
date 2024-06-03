from typing import Dict, List, Literal, Tuple

from app.api.init_services import init_search_client, init_vector_store
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
import json
from azure.search.documents import SearchItemPaged

def search(search_query: str, addVectors: bool, type: Literal['hybrid_search','similarity_search', 'vector_search', 'simple_text'], k, schema_filter: str | None, order_by_date: bool) -> List[Document]:
    client = init_search_client()

    vector_store = init_vector_store()

    if type == 'hybrid_search':
        documents = vector_store.hybrid_search(search_query, k)
    elif type == 'similarity_search':
        documents = vector_store.similarity_search(search_query, k)
    elif type == 'vector_search':
        documents = vector_store.vector_search(search_query, k)
    elif type == 'simple_text':
        result = client.search(search_text=search_query, top=k, order_by='date' if order_by_date else None, filter=f"type eq '{schema_filter}'" if schema_filter is not None else None)
        documents = _results_to_documents(result)

        
    if not addVectors and type != 'simple_text':
        for d in documents:
            d.metadata['content_vector']=[]
    
    return documents


#Gekopieerd uit AzureSearch.py package (via bvb hybrid_search naartoe gegaan)

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
