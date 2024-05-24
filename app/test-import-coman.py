import sys
sys.path.insert(0, "app/coman")
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from coman import ComanLoader
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
import math

load_dotenv()

comanDict = {
    "Actua": "1482edab-dac9-4400-bcd7-ab2dd28b96d2",
    # "Dossiers": "8516a849-55ee-4f7a-ad33-e7c6b089ee8f",
    # "Rechtspraak": "b8c42024-2e29-4e50-b05b-8c888e85f932",
    # "Syllabi": "21340ce4-1459-45c0-983d-8ae7f048fcf0",
    # "Vraag&antwoord": "48a35c82-ed45-4cc2-87b6-cbbd6160f870",
    # "Media": "e1381008-7228-4261-b92b-6d8b4152874a"
}

comanContentSchemeFields = {
    "Actua": [
        "Title",
        "Introduction",
        "Full_text"
    ],
    "Dossiers": [
        "Title",
        "Description",
        "Short_description"
    ],
    "Rechtspraak": [
        "Title",
        "Introduction",
        "Description"
    ],
    "Syllabi": [
        "Title",
        "Short_description"
    ],
    "Vraag&antwoord": [
        "Title",
        "Short_description",
        "Answer",
        "Conclusion"
    ],
    "Media": [
        "Title",
        "Full_text",
        "Introduction",
    ]
}

AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="orisai-text-embedding-3-large-development",
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
        name="sub_type",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="is_public",
        type=SearchFieldDataType.Boolean,
        searchable=False,
        filterable=True,
    ),
    SearchableField(
        name="date",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=False,
        filterable=True,
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
        name="links",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=True,
    )
]

index_name: str = "test-index-coman"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint="https://orisai-search-development.search.windows.net",
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)

for scheme in comanDict:
    loader = ComanLoader.ComanLoader(comanDict[scheme], comanContentSchemeFields[scheme], scheme)

    documents = loader.lazy_load()
    amountDocs = len(documents)
    batchSize = 500
    for i in range(math.ceil(amountDocs / batchSize)):
        print(i)
        start = i * batchSize
        end = ((i + 1) * batchSize) if (i + 1) * batchSize < amountDocs else amountDocs
        print("loading from " + str(start) + " to " + str(end))
        # Upload to azure search with batch size, because otherwise we get an error: Request is too large
        # vector_store.add_documents(documents[start:end])

    print(scheme + " done")
# docs = vector_store.similarity_search("Wanneer wordt Brusselse woonfiscaliteit hervormd?", k=2, filters="source eq '3f37ed58-cd2b-4c76-af56-b1b5fb5a8861'")
