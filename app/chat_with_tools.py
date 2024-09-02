from dotenv import load_dotenv
import os
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from langchain.schema import LLMResult
import uuid
from datetime import datetime
from chat.chat_state import ChatState
from langchain_core.documents import Document
import json
from chat.conversational_tools_executor import create_conversational_tool_executor
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from chat.write_email import generate_email

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

load_dotenv()

credential = AzureNamedKeyCredential(str(os.getenv("AZURE_STORAGE_NAME")), str(os.getenv("AZURE_TABLES_KEY")))
table_service_client = TableServiceClient(
    endpoint=str(os.getenv("AZURE_TABLES_URL")), credential=credential
)
table_client = table_service_client.get_table_client(table_name=str(os.getenv("AZURE_TABLES_CHAT_LOGGING_NAME")))


def log_interaction(question: str, answer: str, prompt: str, documents: List[Document], chain_id: str):
    serializable_documents = []
    for d in documents:
        serializable_documents.append({"page_content": d.page_content, "metadata": d.metadata})

    log_entry = {
            "PartitionKey": "LLMLogs",
            "RowKey": str(uuid.uuid4()),
            "Question": question,
            "Answer": answer,
            "Prompt": prompt,
            "Documents": json.dumps(serializable_documents),
            "Timestamp": datetime.now().isoformat(),
            "ChainId": chain_id
        }
    table_client.create_entity(entity=log_entry)

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        ChatState.prompt = "\n".join(prompts)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        ChatState.answer = response.generations[0][0].text
        log_interaction(ChatState.question, ChatState.answer, ChatState.prompt, ChatState.documents, ChatState.chain_id)


def document_data(conversational_rag_chain: RunnableWithMessageHistory, query):    
    ChatState.question = query
    ChatState.chain_id = str(uuid.uuid4())
    return conversational_rag_chain.invoke(
        {"input": query}, #, "context": ""
        config={"configurable": {"session_id": "abc1234"},"callbacks": [CustomHandler()]},
    )   
    
if __name__ == '__main__':

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]
    if "last_generated_prompt" not in st.session_state:
       st.session_state["last_generated_prompt"]='test'

    st.header("QA ChatBot")
    # ChatInput
    prompt = st.chat_input("Enter your questions here")

    with st.sidebar:
        system_prompt = st.text_area(value="""You are an assistant for question-answering tasks. 
                                     
You can only use the following pieces of retrieved context to answer the question. 
                                     
If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in dutch.
                                     
The date of today is: """ + str(datetime.now()), label="Systeem prompt", height=275
, help="""Eerst wordt gezocht naar de x (hieronder te configureren) best matchende documenten in de vector store. 
Vervolgens wordt deze systeem prompt, samen met de inhoud van die documenten naar de llm gestuurd om een antwoord te genereren
""")
        nr_of_docs_to_retrieve = st.number_input(value=3, label="Aantal documenten die meegestuurd worden", min_value=1,
            help="Aantal documenten die opgehaald worden uit de vector store en meegestuurd worden naar de llm")
        
        score_threshold = st.number_input(value=float(0.7), label="Minimum niveau van zekerheid over document", min_value=float(0), max_value=float(1), step=float(0.01),
            help="Getal tussen 0 en 1 dat aangeeft hoe zeker de vector store minstens moet zijn over een document om het terug te geven. 0 is alle document terug geven ongeacht de zekerheid, 1 zal zo goed als geen enkel document teruggeven")

        context = st.selectbox("Selecteer je context", options=["CIB-lid", "Niet CIB-lid", "Syllabusverbod"], help="""
                     CIB-Lid: Toegang tot alles\n
                     Niet CIB-lid: Enkel toegang tot publieke zaken (geen bijlages)\n
                     Syllabusverbod: Toegang tot alles behalve syllabi\n
                     """)
        
       

    if "user_prompt_history" not in st.session_state:
       st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
       st.session_state["chat_answers_history"]=[]

    if prompt:
       with st.spinner("Generating......"):
            ### Statefully manage chat history ###
            if "store" not in st.session_state:
                st.session_state["store"] = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state["store"]:
                    st.session_state["store"][session_id] = InMemoryHistory()
                # return st.session_state["store"][session_id]
                return InMemoryHistory()
            
            agent=create_conversational_tool_executor(system_prompt,context,nr_of_docs_to_retrieve,score_threshold, get_session_history, prompt)
            res=document_data(query=prompt, conversational_rag_chain=agent)
            print(res)
            answer = res

            # output = res.get("output")
            # # Storing the questions, answers and chat history
            # answer=output['answer']
            # sources=[]
            # for c in output['context']:
            #     if c.metadata['type'] == "Actua":
            #         sources.append(f"[{c.metadata['title']}](https://cib.be/actua/{c.metadata['source']}/blabla)\n")
            #     else:
            #         sources.append(f"[{c.metadata['title']}](https://cib.be/kennis/{c.metadata['source']}/blabla)\n")
            # if(len(sources) > 0):
            #     answer += "\n#### Bronnen:\n" 
            #     answer += "\n".join(sources)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(answer)

            
    with st.sidebar:
        last_generated_prompt_text_area = st.text_area("De prompt die naar de llm is gestuurd", height=275, value=ChatState.prompt)

    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)