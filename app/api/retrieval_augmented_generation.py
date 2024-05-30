from app.api.init_services import init_custom_retriever, init_llm
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def get_filter_for_context(context):
    if context =="CIB_MEMBER":
        return None
    else:
        return "is_public eq 'True'"

def retrieval_augmented_generation(question: str, top_k: int, score_threshold: float, system_prompt: str, context: str, include_page_content: bool):
    filters = get_filter_for_context(context)
    retriever = init_custom_retriever(top_k, filters, score_threshold)
    llm = init_llm()

    ### Answer question ###
    qa_system_prompt = system_prompt + """"

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": question})

    if not include_page_content:
        for d in result["context"]:
            d.page_content = ""

    return result