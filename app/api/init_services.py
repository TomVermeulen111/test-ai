import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.retrievers import BaseRetriever
from app.chat.CustomAzureSearchVectorStoreRetriever import CustomAzureSearchVectorStoreRetriever
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential


def init_vector_store(index_name: str) -> AzureSearch:
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="orisai-text-embedding-3-large-development",
    )

    return AzureSearch(
        azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
        azure_search_key=str(os.getenv("AZURE_SEARCH_KEY")),
        index_name=index_name,
        embedding_function=embeddings.embed_query
    )

def init_retriever(k) -> BaseRetriever:
    index_name=str(os.getenv("AZURE_SEARCH_INDEX_NAME"))

    AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))

    return AzureAISearchRetriever(
        content_key="content", index_name=index_name, service_name="orisai-search-development", api_key=AZURE_SEARCH_KEY, top_k=k
    )

def init_llm() -> AzureChatOpenAI:    
    return AzureChatOpenAI(
        openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
        azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
    )

def init_custom_retriever(k: int, filters: str | None, score_threshold: float, search_type: str = "similarity_score_threshold") -> BaseRetriever:
    vector_store = init_vector_store()
    return CustomAzureSearchVectorStoreRetriever(
        vectorstore=vector_store, 
        k=k, 
        filters=filters, 
        tags=vector_store._get_retriever_tags(),
        search_type=search_type,
        score_threshold=score_threshold
    )

def init_search_client(index_name:str):
    service_endpoint = str(os.getenv("AZURE_SEARCH_BASE_URL"))
    key = str(os.getenv("AZURE_SEARCH_KEY"))
    return SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

def init_table_client(): 
    credential = AzureNamedKeyCredential(str(os.getenv("AZURE_STORAGE_NAME")), str(os.getenv("AZURE_TABLES_KEY")))
    table_service_client = TableServiceClient(
        endpoint=str(os.getenv("AZURE_TABLES_URL")), credential=credential
    )
    return table_service_client.get_table_client(table_name=str(os.getenv("AZURE_TABLES_CHAT_LOGGING_NAME")))