import os
from chat.conversational_rag_chain import create_conversational_rag_chain
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain.evaluation import Criteria
from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from langchain_openai import AzureChatOpenAI
import csv
import uuid

load_dotenv()

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return InMemoryHistory()

chain=create_conversational_rag_chain("""You are an assistant for question-answering tasks. 
                                     
You can only use the following pieces of retrieved context to answer the question. 
                                     
If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question
                                     
Use three sentences maximum and keep the answer concise.
                                     
You will have a chat history, but you must only answer the last question.
                                     
You MUST answer in dutch.
                                     
The date of today is: """ + str(datetime.now()),"CIB-lid",3,0.7, get_session_history)

result = chain.invoke(
        {"input": 'wat is er de voorbije jaren allemaal veranderd in de brusselse woningfiscaliteit?'},
        config={"configurable": {"session_id": "abc123"}},
    )  
answer = result['answer']

llm=AzureChatOpenAI(
        openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
        azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
    )

criteria_list = [Criteria.CORRECTNESS, Criteria.CONCISENESS]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(f'evaluation\evaluation_results\{timestamp}')

for criteria in criteria_list:
    evaluator = load_evaluator(EvaluatorType.LABELED_CRITERIA, llm=llm, criteria=criteria)
    results = []
    with open('evaluation\evaluation_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            question=row[0]
            expexted_answer=row[1]
            chain_result = chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": str(uuid.uuid4())}},
            )
            answer = chain_result['answer']
            answer_context = chain_result['context']
            eval_result = evaluator.evaluate_strings(
                prediction=answer,
                input=question,
                reference=expexted_answer
            )
            results.append({"question": question, "expected_answer": expexted_answer, "answer": answer, "answer_context": answer_context, "eval_result": eval_result})
    
    with open(f'evaluation\evaluation_results\{timestamp}\{criteria}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", "expected_answer", "actual_answer", "answer_context", "passed_evaluation?", "evaluation_score", "evaluation_reasoning"])
        for result in results:
            writer.writerow([result["question"], result["expected_answer"], result["answer"], result["answer_context"], result["eval_result"]["value"], result["eval_result"]["score"], result["eval_result"]["reasoning"]])