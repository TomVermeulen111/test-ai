import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.retrievers import BaseRetriever
from app.chat.CustomAzureSearchVectorStoreRetriever import CustomAzureSearchVectorStoreRetriever
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents import SearchClient

def init_vector_store() -> AzureSearch:
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="orisai-text-embedding-3-large-development",
    )

    return AzureSearch(
        azure_search_endpoint=str(os.getenv("BASE_URL")),
        azure_search_key=str(os.getenv("AZURE_SEARCH_KEY")),
        index_name=str(os.getenv("INDEX_NAME")),
        embedding_function=embeddings.embed_query
    )

def init_retriever(k) -> BaseRetriever:
    index_name=str(os.getenv("INDEX_NAME"))

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

def init_search_client():
    return SearchClient(
        endpoint=str(os.getenv("BASE_URL")), 
        api_key=str(os.getenv("AZURE_SEARCH_KEY")), 
        index_name=str(os.getenv("INDEX_NAME"))
    )