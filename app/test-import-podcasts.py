import sys

from loaders import PodcastLoader
from chat.coman_schemes import ComanScheme
sys.path.insert(0, "app/loaders")
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
import math
import datetime
import time


load_dotenv()

AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=str(os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")),
)

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embeddings.embed_query("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchableField(
        name="source",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="type",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
]

index_name: str = "podcasts-index"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)

print('start loading: ', datetime.datetime.now())
loader = PodcastLoader.PodcastLoader("C:\\Users\\TomVermeulen\\OneDrive - OPEN REAL ESTATE INFORMATION SERVICES AFGEKORT ORIS\\Documents\\Podcasts\\Vastgoedactueel", "C:\\Users\\TomVermeulen\\OneDrive - OPEN REAL ESTATE INFORMATION SERVICES AFGEKORT ORIS\\Documents\\Podcasts\\Vastgoedpraat")

documents = loader.lazy_load()
amountDocs = len(documents)
# print("Amount of docs: " + str(amountDocs))
batchSize = 500
print('total amount of docs to load: ', amountDocs)
for i in range(math.ceil(amountDocs / batchSize)):
    # print(i)
    start = i * batchSize
    end = ((i + 1) * batchSize) if (i + 1) * batchSize < amountDocs else amountDocs

    print("loading from " + str(start) + " to " + str(end))
    # Upload to azure search with batch size, because otherwise we get an error: Request is too large
    vector_store.add_documents(documents[start:end])
    time.sleep(2)

