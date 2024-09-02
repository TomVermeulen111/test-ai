import os
from typing import Dict, List, Literal, Tuple
from app.api.filters import SearchFilters, SortOrder, SortField
from app.api.init_services import init_search_client, init_vector_store
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
import json
from azure.search.documents import SearchItemPaged
from azure.search.documents import SearchClient
from multiprocessing import Process

class SearchResultItem():
    document: Document
    highlights: dict
    score: float

    def __init__(self, document: Document, highlights: dict, score: float):
        self.document = document
        self.highlights = highlights
        self.score = score

class SearchResult():
    results: List[SearchResultItem]
    count: int

    def __init__(self, results: List[SearchResultItem], count: int):
        self.results = results
        self.count = count


def search(
        search_query: str, 
        addVectors: bool, 
        type: Literal['hybrid_search','similarity_search', 'vector_search', 'simple_text'], 
        order_by_date: bool,
        filters: SearchFilters,
        top: int,
        skip: int) -> SearchResult:
    cib_client = init_search_client(str(os.getenv("AZURE_SEARCH_INDEX_NAME")))
    model_docs_client = init_search_client(str(os.getenv("AZURE_SEARCH_MODELDOCS_INDEX_NAME")))

    coman_vector_store = init_vector_store(str(os.getenv("AZURE_SEARCH_INDEX_NAME")))
    modeldocs_vector_store = init_vector_store(str(os.getenv("AZURE_SEARCH_MODELDOCS_INDEX_NAME")))

    filter=get_filter(filters)
    searchResult = SearchResult(count=0, results=[])

    if type == 'hybrid_search':
        searchResult.documents = coman_vector_store.hybrid_search(search_query, top)
        searchResult.documents.extend(modeldocs_vector_store.hybrid_search(search_query, top))
    elif type == 'similarity_search':
        searchResult.documents = coman_vector_store.similarity_search(search_query, top)
        extra_docs = modeldocs_vector_store.similarity_search(search_query, top)
        searchResult.documents.extend(extra_docs)
    elif type == 'vector_search':
        searchResult.documents = coman_vector_store.vector_search(search_query, top)
    elif type == 'simple_text':
        cib_results = SearchResult(results=[], count=0)
        model_docs_results = SearchResult(results=[], count=0)
        fns = []
        if (filter is None or filter == '' or 'Modeldocumenten' in filter):
            fns.append(simple_search(model_docs_results, model_docs_client, search_text=search_query, order_by='date' if order_by_date else None, filter=None, top=100, skip=0, query_type="simple", highlight_fields="content", highlight_pre_tag='<b>', highlight_post_tag='</b>', include_total_count=True))
        
        if (filter is None or filter == '' or (filter != '' and filter != 'type eq \'Modeldocumenten\'')):
            fns.append(simple_search(cib_results, cib_client, search_text=search_query, order_by='date' if order_by_date else None, filter=filter, top=100, skip=0, query_type="simple", highlight_fields="content", highlight_pre_tag='<b>', highlight_post_tag='</b>', include_total_count=True))

        runInParallel(*fns)
        cib_results.results.extend(model_docs_results.results)

        single_results = []
        
        for result in cib_results.results:
            if (any(res.document.metadata['source'] == result.document.metadata['source'] for res in single_results)):
                continue
            single_results.append(result)
        searchResult.count = len(single_results)
        searchResult.results = sorted(single_results, key=lambda x: x.score if (filters.sortingFilter.field.value == SortField.SCORE.value) else x.document.metadata['date'], reverse=(filters.sortingFilter.order.value == SortOrder.DESC.value))
        searchResult.results = searchResult.results[skip:top+skip]
        
    if not addVectors and type != 'simple_text':
        for d in searchResult.documents:
            d.metadata['content_vector']=[]
    return searchResult

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


def simple_search(result: SearchResult, client: SearchClient, search_text, order_by, filter, top, skip, query_type, highlight_fields, highlight_pre_tag, highlight_post_tag, include_total_count):
    results = client.search(search_text=search_text, order_by=order_by, filter=filter, top=top, skip=skip, query_type=query_type, highlight_fields=highlight_fields, highlight_pre_tag=highlight_pre_tag, highlight_post_tag=highlight_post_tag, include_total_count=include_total_count)
    result.count = results.get_count()
    for r in results:
        result.results.append(
            SearchResultItem(
                document=_result_to_document(r), 
                highlights=r['@search.highlights'], 
                score=r['@search.score']
            )
        )

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
) -> List[Tuple[Document, Dict]]:
    docs = []
    
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


def get_filter(filters: SearchFilters) -> str:
    filterItems = []
    excludeTypes = [type.value for type in filters.excludeTypeFilter]
    excludeFilterItems = []
    for type in excludeTypes:
        excludeFilterItems.append(f"type ne '{type}'")
    filterItems.append(" and ".join(excludeFilterItems))
    
    if filters.domainFilter is not None and len(filters.domainFilter) > 0:
        domains = [domain.value for domain in filters.domainFilter]
        domainFilterItem = f"domains/any(domain: search.in(domain, '{','.join(domains)}', ','))"
        filterItems.append(domainFilterItem)
        
    if filters.categoryFilter is not None and len(filters.categoryFilter) > 0:
        categories = [category.value for category in filters.categoryFilter]
        categoryFilterItem = f"categories/any(category: search.in(category, '{','.join(categories)}', ','))"
        filterItems.append(categoryFilterItem)
        
    if filters.typeFilter is not None and len(filters.typeFilter) > 0:
        types = [type.value for type in filters.typeFilter]
        typeFilterItems = []
        for type in types:
            typeFilterItems.append(f"type eq '{type}'")
        typeFilter = "(" + " or ".join(typeFilterItems) + ")"
        filterItems.append(typeFilter)
    filter = " and ".join(filterItems)
    return filter