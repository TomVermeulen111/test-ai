import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from chat.CustomAzureSearchVectorStoreRetriever import CustomAzureSearchVectorStoreRetriever
from chat.write_email import generate_email
from langchain.tools.render import render_text_description
from langchain.agents import create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.chains.base import Chain
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, RetrieverInput
from functools import partial

def create_conversational_tool_executor(system_prompt, context, nr_of_docs_to_retrieve, score_threshold, get_session_history, prompt):   
    # https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/

    index_name=str(os.getenv("AZURE_SEARCH_INDEX_NAME"))

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
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=str(os.getenv("AZURE_SEARCH_BASE_URL")),
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
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    print(prompt)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    retriever_tool = create_custom_chain_tool(
        rag_chain,
        "search",
        "Search for information about anything related to real estate. For any questions about real estate, CIB, you MUST use this tool! Also, if no other appropriate tool is found, use this tool",
        prompt
        # document_prompt=qa_prompt
    )
    print(retriever_tool)

    retriever_tool = StructuredTool.from_function()
    tools = [generate_email, retriever_tool]
    
    # TODO Check: is prompt correct here? Not a ChatPromptTemplate?
    agent = create_tool_calling_agent(llm, tools, qa_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    
    return agent_with_chat_history


def create_custom_chain_tool(chain: Chain, name: str, description: str, prompt: str) -> BaseTool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    def custom_chain_func(input_text: str, chat_history: list, context: dict) -> Any:
        return chain.invoke({
            "input": input_text,
            "chat_history": chat_history,
            "context": context
        })
    
    # document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    print(chain)
    return Tool(
        name=name,
        description=description,
        func=chain.invoke({"input": prompt, "chat_history": []}, config={"configurable": {"session_id": "abc123"}}),
        func = partial(
            chain.invoke,
            input={"input": prompt, "chat_history": []}, config={"configurable": {"session_id": "abc123"}}
        ),
        # func=custom_chain_func,
        args_schema=RetrieverInput,
    )