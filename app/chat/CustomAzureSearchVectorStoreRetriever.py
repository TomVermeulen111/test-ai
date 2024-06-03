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

from chat_state import ChatState

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
        if self.search_type == "similarity":
            docs = self.vectorstore.vector_search(query, k=self.k,**kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.k, **kwargs
                )
            ]
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=self.k, **kwargs)
        elif self.search_type == "hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.hybrid_search_with_relevance_scores(
                    query, k=self.k, **kwargs
                )
            ]
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k, **kwargs)
        elif self.search_type == "semantic_hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.semantic_hybrid_search_with_score(
                    query, k=self.k, **kwargs
                )
            ]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
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