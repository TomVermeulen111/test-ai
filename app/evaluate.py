import os
from chat.conversational_rag_chain import create_conversational_rag_chain
from datetime import datetime
from dotenv import load_dotenv
from langchain.evaluation import Criteria
from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from langchain_openai import AzureChatOpenAI
import csv
import uuid

load_dotenv()

#The params for the evaluation run
file_name="Prompt-v2"

nr_of_docs_to_retrieve=3 
score_threshold=0.7
criteria_list = [Criteria.CORRECTNESS, Criteria.RELEVANCE]
# app\evaluation\evaluation_data_high_med_prio.csv
# app\evaluation\evaluation_data_high_prio.csv
evaluation_data_path = "app\evaluation\evaluation_data_high_med_prio.csv"
# score_increases_per_type={"Syllabi": 0.02, "Rechtspraak": 0.02}
score_increases_per_type={}
system_prompt="""Act as a professional assistant for CIB that answers questions of our members who are mainly realestate brokers and syndics.
Your instructions are to help the CIB-members with all their questions, from general questions, questions about CIB organization, online tools, juridical questions etc.
The end goal is that the conversation partner is well informed and doesn't need to ask the question to a human (legal) expert in real-estate.
You can only use the following pieces of retrieved context to answer the question.
If you cannot answer the question with the provided context or there is no context provided, inform the user that you do not have enough information to answer the question.
If you find multiple answers or if your answer would be too generic, ask the user to specify his question more. 
Indicate where he needs to specify.
Use four sentences maximum and keep the answer concise and don't use overly flawed language.
You will have a chat history, but you must only answer the last question.
You MUST answer in dutch.
The date of today is:  """ + str(datetime.now())


chain=create_conversational_rag_chain(
    system_prompt=system_prompt,
    nr_of_docs_to_retrieve=nr_of_docs_to_retrieve,
    score_threshold=score_threshold,
    score_increases_per_type=score_increases_per_type
)

llm=AzureChatOpenAI(
    openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
    azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
)

#evaluate the answer, sometimes the llm doesnt format it's answer correctly, so we retry a few times
def evaluate(evaluator, answer, question, expexted_answer, retry_count=0):
    try:
        if(retry_count > 5):
            print("Failed to evaluate after 5 retries, skipping...")
            return {"score": 0, "reasoning": "Failed to evaluate after 5 retries"}

        return evaluator.evaluate_strings(
            prediction=answer,
            input=question,
            reference=expexted_answer
        )
    except ValueError:
        print("Answer provided by llm was formatted incorrectly, trying again...")
        return evaluate(evaluator, answer, question, expexted_answer, retry_count+1)

results = []
#load evaluation data
with open(evaluation_data_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='|')
    line_count = 0

    #iterate each row in the evaluation data
    for row in csv_reader:
        question=row[0]
        expexted_answer=row[1]
        chain_result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": str(uuid.uuid4())}},
        )
        answer = chain_result['answer']
        answer_context = chain_result['context']
        currentResult = {"question": question, "expected_answer": expexted_answer, "answer": answer, "answer_context": answer_context}

        #evaluate the answer with each criteria
        for criteria in criteria_list:
            evaluator = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, llm=llm, criteria=criteria)
            print("evaluating row", line_count, " with criteria", criteria.value)
            eval_result = evaluate(evaluator, answer, question, expexted_answer)
            currentResult["eval_result_"+criteria] = eval_result

        line_count += 1
        results.append(currentResult)
    
    
#write results to csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'app\evaluation\evaluation_results\{file_name}-{timestamp}.csv', 'w', newline='', errors='ignore') as file:
    writer = csv.writer(file)
    headings = ["question", "expected_answer", "actual_answer", "answer_context"]
    
    for criteria in criteria_list:
        headings.append("evaluation_score_"+criteria)
        headings.append("evaluation_reasoning_"+criteria)

    headings.append("average_score")

    #write headings
    writer.writerow(headings)

    #for each result, write the question, expected answer, actual answer, answer context, evaluation score and reasoning (for each criteria) and average score
    for result in results:
        row = [result["question"], result["expected_answer"], result["answer"], result["answer_context"]]
        for criteria in criteria_list:
            eval_result = result["eval_result_"+criteria]
            row.append(eval_result["score"])
            row.append(eval_result["reasoning"])

        row.append(sum([result["eval_result_"+criteria]["score"] for criteria in criteria_list])/len(criteria_list))

        writer.writerow(row)

    #write the used params to evaluate
    metadata = f"nr_of_docs_to_retrieve={nr_of_docs_to_retrieve},score_threshold={score_threshold},evaluation_data_path={evaluation_data_path},score_increases_per_type={score_increases_per_type},system_prompt={system_prompt}"
    writer.writerow(["metadata", metadata])
