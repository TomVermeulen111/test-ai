# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import ShellTool

class EmailInput(BaseModel):
    email_input: str = Field(description="should be the content of the email."),
    userName: str = Field(description="should be the name of the sender.")
 

@tool("write_email", args_schema=EmailInput, return_direct=True)
def write_email(email_input: str, userName: str) -> str:
    """Useful when you need to generate an email based on the response or rewrite a response as an email."""
    print("Writing email now")
    return f"""Beste,
    Via deze email willen we u informeren over:
    {email_input}

    Met vriendelijke groeten,
    {userName}
    """
