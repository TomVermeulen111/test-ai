from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import uuid
from datetime import datetime
from chat.chat_state import ChatState
from langchain_core.documents import Document
import json
from chat.conversational_rag_chain import create_conversational_rag_chain
from langchain_core.chat_history import BaseChatMessageHistory
from api.init_services import init_table_client

def chat(question: str, chat_history: BaseChatMessageHistory):
    
    chain=create_conversational_rag_chain(
        get_session_history=lambda session_id: chat_history
    )

    ChatState.question = question
    ChatState.chain_id = str(uuid.uuid4())

    return chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "abc-123"},"callbacks": [CustomHandler()]},
    )   

def log_interaction(question: str, answer: str, prompt: str, documents: List[Document], chain_id: str):
    table_client = init_table_client()

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