import os
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.retrievers import BaseRetriever

def init_retriever() -> BaseRetriever:
    index_name=str(os.getenv("INDEX_NAME"))

    AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))

    return AzureAISearchRetriever(
        content_key="content", index_name=index_name, service_name="orisai-search-development", api_key=AZURE_SEARCH_KEY, top_k=1
    )

def init_llm() -> AzureChatOpenAI:    
    return AzureChatOpenAI(
        openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
        azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
    )