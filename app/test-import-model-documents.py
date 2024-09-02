import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_transformers import Html2TextTransformer
from loaders import ModelDocLoader
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
        name="lastPublicationDocumentId",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="lastPublicationVersionComment",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="lastPublicationVersion",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="locationRealEstate",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="code",
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
    SearchableField(
        name="date",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=False,
        filterable=True,
        sortable=True,
    ),
    SearchableField(
        name="categories",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="domains",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="language",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="document_id",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="document_name",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="document_link",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="document_type",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="chunk_number",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="page",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    )
]

index_name: str = os.getenv("AZURE_SEARCH_MODELDOCS_INDEX_NAME")
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)

html2text = Html2TextTransformer()

print('start loading: ', datetime.datetime.now())
loader = ModelDocLoader.ModelDocLoader()

documents = loader.lazy_load()
# Transform the document (html to text, skip images, etc.)
documents = html2text.transform_documents(documents)
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

print('done loading modeldocs: ', datetime.datetime.now())
# docs = vector_store.similarity_search("Wanneer wordt Brusselse woonfiscaliteit hervormd?", k=2, filters="source eq '3f37ed58-cd2b-4c76-af56-b1b5fb5a8861'")
