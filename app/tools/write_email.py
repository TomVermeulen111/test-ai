# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import ShellTool

class EmailInput(BaseModel):
    email_input: str = Field(description="should be the content of the email."),
    name: str = Field(description="should be the name of the sender.")
 

@tool("write-email", args_schema=EmailInput, return_direct=True)
def write_email(email_input: str, name: str) -> str:
    """useful when you need to generate an email based on the response."""
    print("Writing email now")
    return f"""Beste,
    Via deze email willen we u informeren over:
    ${email_input}

    Met vriendelijke groeten,
    {name}
    """


# Define the email tool
# class WriteEmailTool(StructuredTool):
#     response: str = Field(...)

#     def run(self) -> str:
#         """Write an email to the user based on the response."""
#         print("Writing email now")
#         return f"""Beste,
#         Via deze email willen we u informeren over:
#         {self.response}
#         """