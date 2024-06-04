import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from CustomAzureSearchVectorStoreRetriever import CustomAzureSearchVectorStoreRetriever
from write_email import write_email
from langchain.tools.render import render_text_description

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

        
def create_conversational_rag_chain(system_prompt, context, nr_of_docs_to_retrieve, score_threshold):   
    # https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

    index_name=str(os.getenv("INDEX_NAME"))

    llm = AzureChatOpenAI(
        openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
        azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
    )

    AZURE_SEARCH_KEY = str(os.getenv("AZURE_SEARCH_KEY"))

    def get_filter_for_context(context):
        if context =="CIB-lid":
            return None
        elif context == "Niet CIB-lid":
            return "is_public eq 'True'"
        elif context == "Syllabusverbod":
            return "type ne 'Syllabi'"
    
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).with_config()
   
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="orisai-text-embedding-3-large-development",
    )

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=str(os.getenv("BASE_URL")),
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=index_name,
        embedding_function=embeddings.embed_query
    )

    retriever = CustomAzureSearchVectorStoreRetriever(
        vectorstore=vector_store, 
        k=nr_of_docs_to_retrieve, 
        filters=get_filter_for_context(context), 
        tags=vector_store._get_retriever_tags(),
        search_type="similarity_score_threshold",
        score_threshold=score_threshold
    )

    rendered_tools = render_text_description([write_email])

    system_prompt = f"""
You have access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the result of the tool to use.
When a tool is available for a specific task, DO NOT ANSWER THE QUESTION YOURSELF BUT USE THE TOOL INSTEAD AND RETURN ITS RESULT!
    """ + system_prompt
                                             
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = system_prompt + """"
    <context>
    {context}
    </context>"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    if "store" not in st.session_state:
        st.session_state["store"] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state["store"]:
            st.session_state["store"][session_id] = InMemoryHistory()
        return st.session_state["store"][session_id]
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain
