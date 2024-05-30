import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="orisai-text-embedding-3-large-development",
)


index_name: str = "test-index"


vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint="https://orisai-search-development.search.windows.net",
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

loader = PyPDFLoader("app/data/ticket.PDF")

docs = loader.load()


doc = docs[0]

doc.metadata["foo"] = "bar"
print(doc)

foo = [doc]


vector_store.add_documents(documents=foo)


db = vector_store

# db = vector_store

query = "what is the secret ingredient of the recipe?"
docs = db.similarity_search(query, k=1)

# print(docs)
for d in docs:
    print(d.metadata)

# print(docs)

