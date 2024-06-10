import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
from vivo import VivoLoader
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
import datetime
import time
import math

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
        name="course_id",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_title",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_knowledge_level",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_prior_knowledge_level",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_language",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_learning_form",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="course_target_groups",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    )
]

index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME")
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)

html2text = Html2TextTransformer()

print('start loading: ', datetime.datetime.now())

loader = VivoLoader.VivoLoader()
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

print('done loading: ', datetime.datetime.now())