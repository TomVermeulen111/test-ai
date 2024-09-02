import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from chat.CustomAzureSearchVectorStoreRetriever import CustomAzureSearchVectorStoreRetriever
from langchain.tools.render import render_text_description
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory
from chat.write_email import generate_email
from langchain.retrievers import EnsembleRetriever
        
def create_conversational_rag_chain(
        system_prompt="""You are an assistant for question-answering tasks. 
                                        
    You can only use the following pieces of retrieved context to answer the question.
                                        
    If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question
                                        
    Use three sentences maximum and keep the answer concise.
                                        
    You will have a chat history, but you must only answer the last question.
                                        
    You MUST answer in dutch.
                                        
    The date of today is: """ + str(datetime.now()), 
        context="CIB-lid", 
        nr_of_docs_to_retrieve=3, 
        score_threshold=0.7, 
        get_session_history=lambda session_id: InMemoryChatMessageHistory(),
        score_increases_per_type={"Syllabi": 0.02, "Rechtspraak": 0.02}
    ):   
    # https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

    coman_index_name=str(os.getenv("AZURE_SEARCH_INDEX_NAME"))
    modeldocs_index_name=str(os.getenv("AZURE_SEARCH_MODELDOCS_INDEX_NAME"))

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
            ("system", "reminder: do not answer the question, just reformulate it if needed")
        ]
    ).with_config()
   
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )

    coman_vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=coman_index_name,
        embedding_function=embeddings.embed_query
    )

    modeldocs_vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=modeldocs_index_name,
        embedding_function=embeddings.embed_query
    )

    retriever = CustomAzureSearchVectorStoreRetriever(
        vectorstore=coman_vector_store,
        k=nr_of_docs_to_retrieve, 
        filters=get_filter_for_context(context), 
        tags=coman_vector_store._get_retriever_tags(),
        search_type="similarity_score_threshold",
        score_threshold=score_threshold,
        score_increase_per_type=score_increases_per_type
    )

    modeldocs_retriever = CustomAzureSearchVectorStoreRetriever(
        vectorstore=modeldocs_vector_store, 
        k=nr_of_docs_to_retrieve, 
        filters=get_filter_for_context(context), 
        tags=coman_vector_store._get_retriever_tags(),
        search_type="similarity_score_threshold",
        score_threshold=score_threshold
    )

#     rendered_tools = render_text_description([generate_email])

#     system_prompt = f"""
# You have access to the following set of tools. Here are the names and descriptions for each tool:

# {rendered_tools}

# Given the user input, return the result of the tool to use.
# When a tool is available for a specific task, DO NOT ANSWER THE QUESTION YOURSELF BUT USE THE TOOL INSTEAD AND RETURN ITS RESULT!
#     """ + system_prompt

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever, modeldocs_retriever], weights=[0.5, 0.5])

    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """
    <context>
    {context}
    </context>
    """ + system_prompt

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain
