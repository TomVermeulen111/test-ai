# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import ShellTool

class EmailInput(BaseModel):
    email_input: str = Field(description="The content of the email to be sent."),
    sender: str = Field(description="The name of the sender."),
    email_address: str = Field(description="The email address of the receiver."),
    receiver: str = Field(description="should be the name of the receiver."),
    subject: str = Field(description="The subject of the email."),
 

# @tool("write_email", args_schema=EmailInput, return_direct=False, infer_schema=True)
# def write_email(email_input: str, sender: str, email_address: str, receiver: str, subject: str) -> str:
#     """Tool that HAS to be used when you need to generate an email based on the response or rewrite a response as an email.
#     A chatbot SHOULD ask for the necessary parameters to generate the email.

#     Args:
#         email_input: The content of the email to be sent.
#         sender: The name of the sender.
#         email_address: The email address of the receiver.
#         receiver: The name of the receiver.
#         subject: The subject of the email.
    
#     Returns:
#         A string representing the email.

#     """
#     print("Writing email now")
#     return f"""
#     Onderwerp: {subject}
#     To: {receiver} <{email_address}>
#     From: {sender}

#     Beste,
#     Via deze email willen we u informeren over:
#     {email_input}

#     Met vriendelijke groeten,
#     {sender}
#     """

@tool()
def generate_email(input: str) -> str:
    """Useful when you need to generate an email based on the response or rewrite a response as an email. DON'T REWRITE THE OUTPUT OF THIS TOOL!"""
    print("Writing email now")
    return f"""Beste [FILL IN],
    Via deze email willen we u informeren over:
    {input}

    Met vriendelijke groeten,
    [FILL IN]
    """
