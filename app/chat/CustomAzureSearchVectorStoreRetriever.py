from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    List,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.azuresearch import AzureSearch

from chat.coman_schemes import ComanScheme
from chat.chat_state import ChatState

class CustomAzureSearchVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure Cognitive Search`."""

    vectorstore: AzureSearch
    """Azure Search instance used to find similar documents."""
    search_type: str = "hybrid"
    """Type of search to perform. Options are "similarity", "hybrid",
    "semantic_hybrid", "similarity_score_threshold", "hybrid_score_threshold", 
    or "semantic_hybrid_score_threshold"."""
    k: int = 4
    """Number of documents to return."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "hybrid",
        "hybrid_score_threshold",
        "semantic_hybrid",
        "semantic_hybrid_score_threshold",
    )
    """A key-value pair with the key being the type and the value being a number between 0 and 1 with which the score for all retrieved documents of that type will be increased"""
    score_increase_per_type: Dict[str, float] = {}

    filters: str | None = None

    score_threshold: float | None = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in cls.allowed_search_types:
                raise ValueError(
                    f"search_type of {search_type} not allowed. Valid values are: "
                    f"{cls.allowed_search_types}"
                )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        kwargs['filters'] = self.filters
        kwargs['score_threshold'] = self.score_threshold
        #We use a high k to get (almost) all of the docs that are above the threshold
        #Later we will sort them on date and return the top k
        high_k = 99
        if self.search_type == "similarity":
            docs = self.vectorstore.vector_search(query, k=high_k,**kwargs)
        elif self.search_type == "similarity_score_threshold":
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=high_k, **kwargs)
            results.sort(key=lambda x: x[1] + self.score_increase_per_type[x[0].metadata['type']] if x[0].metadata['type'] in self.score_increase_per_type else x[1], reverse=True)
            docs = [doc for doc, _ in results]
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=high_k, **kwargs)
        elif self.search_type == "hybrid_score_threshold":
            results = self.vectorstore.hybrid_search_with_relevance_scores(query, k=high_k, **kwargs)
            results.sort(key=lambda x: x[1] + self.score_increase_per_type[x[0].metadata['type']] if x[0].metadata['type'] in self.score_increase_per_type else x[1], reverse=True)
            docs = [doc for doc, _ in results]
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=high_k, **kwargs)
        elif self.search_type == "semantic_hybrid_score_threshold":
            results = self.vectorstore.semantic_hybrid_search_with_score(query, k=high_k, **kwargs)
            results.sort(key=lambda x: x[1] + self.score_increase_per_type[x[0].metadata['type']] if x[0].metadata['type'] in self.score_increase_per_type else x[1], reverse=True)
            docs = [doc for doc, _ in results]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        #sorts the docs with the specified types by date while preserving the position of the other docs
        docs = self.sort_with_date_relevancy(docs, [ComanScheme.ACTUA.value, ComanScheme.JURISDICTION.value, ComanScheme.MEDIA.value])
        #return the k most relevant docs
        docs = docs[:self.k]

        docs = self.add_date_info_to_page_content(docs)
                
        ChatState.documents = docs
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError(
            "AzureSearchVectorStoreRetriever does not support async"
        )
        
    def sort_with_date_relevancy(self, docs: List[Document], sorted_types: List[str]):
        """Sort objects with the given types by their date property while retaining the original order of the other objects"""
        
        # Get objects that should be sorted
        sortable = [(i, doc) for i, doc in enumerate(docs) if doc.metadata['type'] in sorted_types]
        
        # Sort sortable objects by their date property
        sortable_sorted = sorted(sortable, key=lambda x: x[1].metadata['date'], reverse=True)

        # Merge sorted and non-sorted objects back into the original list
        result_docs = docs[:]
        sortable_index = 0

        for i, result_doc in enumerate(result_docs):
            if result_doc.metadata['type']  in sorted_types:
                result_docs[i] = sortable_sorted[sortable_index][1]
                sortable_index += 1

        return result_docs
    

    def add_date_info_to_page_content(self, docs: List[Document]) -> List[Document]:
        """Add date information to the page content of the documents"""
        for i, doc in enumerate(docs):
            if 'date' in doc.metadata:
                doc.page_content = f"<bron{i+1}>\nDatum van de bron: {doc.metadata['date']}\n{doc.page_content}</bron{i+1}>\n"
        return docs