import sys

from chat.coman_schemes import ComanScheme
sys.path.insert(0, "app/loaders")
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
from loaders import ComanLoader, ComanCollaboratorLoader
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

comanDict = {
    ComanScheme.ACTUA.value: "1482edab-dac9-4400-bcd7-ab2dd28b96d2",
    ComanScheme.DOSSIERS.value: "8516a849-55ee-4f7a-ad33-e7c6b089ee8f",
    # ComanScheme.JURISDICTION.value: "b8c42024-2e29-4e50-b05b-8c888e85f932",
    ComanScheme.SYLLABI.value: "21340ce4-1459-45c0-983d-8ae7f048fcf0",
    # ComanScheme.QUESTION_ANSWER.value: "48a35c82-ed45-4cc2-87b6-cbbd6160f870",
    # ComanScheme.MEDIA.value: "e1381008-7228-4261-b92b-6d8b4152874a",
    # ComanScheme.WEBTEXTS.value: "24230396-013a-442d-92dd-cfb6338f8203",
    # ComanScheme.DEPARTMENTS.value: "6fbe435f-d2ce-415a-93d9-f7f83b28db25",
    # ComanScheme.TOOLS.value: "15d6b179-6fb0-4b25-b4f5-afb066dcc6df",
    # ComanScheme.EVENTS.value: "b47061f1-3cec-4c98-8fa6-5d012ee61dc1",
}

comanContentSchemeFields = {
    ComanScheme.ACTUA.value: [
        "Title",
        "Introduction",
        "Full_text"
    ],
    ComanScheme.DOSSIERS.value: [
        "Title",
        "Description",
        "Short_description"
    ],
    ComanScheme.JURISDICTION.value: [
        "Title",
        "Introduction",
        "Description"
    ],
    ComanScheme.SYLLABI.value: [
        "Title",
        "Short_description"
    ],
    ComanScheme.QUESTION_ANSWER.value: [
        "Title",
        "Short_description",
        "Answer",
        "Conclusion"
    ],
    ComanScheme.MEDIA.value: [
        "Title",
        "Full_text",
        "Introduction",
    ],
    ComanScheme.WEBTEXTS.value: [
        "Title",
        "Long_description"
    ],
    ComanScheme.DEPARTMENTS.value: [
        "Name",
        "Description",
        "Address",
        "Telephone",
        "Email",
        "Member_consulent",
        "Member_consulent_email",
        "Member_consulent_telephone"
    ],
    ComanScheme.TOOLS.value: [
        "Name",
        "Description",
        "Description_advantage",
        "Full_description_tool",
    ],
    ComanScheme.EVENTS.value: [
        "Date",
        "Description",
        "Name",
    ],
}

comanImageSchemeFields = {
    ComanScheme.ACTUA.value: "Image",
    ComanScheme.DOSSIERS.value: "Image",
    ComanScheme.JURISDICTION.value: "Image",
    ComanScheme.SYLLABI.value: "Image",
    ComanScheme.QUESTION_ANSWER.value: "Image",
    ComanScheme.MEDIA.value: "Image",
    ComanScheme.WEBTEXTS.value: "Photo",
    ComanScheme.DEPARTMENTS.value: "Logo",
    ComanScheme.TOOLS.value: "Logo",
    ComanScheme.EVENTS.value: "Picture",
}

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
        name="sub_type",
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
        sortable=True,
    ),
    SearchField(
        name="categories",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        searchable=False,
        filterable=True,
    ),
    SearchField(
        name="domains",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
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
    ),
    SearchableField(
        name="image",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=False,
    ),
    SearchableField(
        name="link",
        type=SearchFieldDataType.String,
        searchable=False,
        filterable=False,
    )
]

index_name: str = str(os.getenv("AZURE_SEARCH_INDEX_NAME"))
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)

html2text = Html2TextTransformer()

print('start loading: ', datetime.datetime.now())
for scheme in comanDict:
    print("start loading " + scheme + ": ", datetime.datetime.now())
    loader = ComanLoader.ComanLoader(comanDict[scheme], comanContentSchemeFields[scheme], comanImageSchemeFields[scheme], scheme)

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

    print(scheme + " done")
    print('done loading ' + scheme + ': ', datetime.datetime.now())


# people = ComanCollaboratorLoader.ComanCollaboratorLoader().lazy_load()
# # vector_store.add_documents(people)
# print(people)
# docs = vector_store.similarity_search("Wanneer wordt Brusselse woonfiscaliteit hervormd?", k=2, filters="source eq '3f37ed58-cd2b-4c76-af56-b1b5fb5a8861'")
