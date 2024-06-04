from langchain.tools.render import render_text_description
from write_email import write_email
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

rendered_tools = render_text_description([write_email])

print(rendered_tools)

system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

model = AzureChatOpenAI(
    openai_api_version=str(os.getenv("AZURE_OPENAI_API_VERSION")),
    azure_deployment=str(os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")),
)

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | write_email
result = chain.invoke({"input": "Bij een gesplitste aankoop wordt de eigendom van een onroerend goed verdeeld tussen de ouders en de kinderen. De ouders kopen het vruchtgebruik, wat hen het recht geeft om het pand te gebruiken en de vruchten ervan te ontvangen, terwijl de kinderen de blote eigendom kopen, wat hen het recht geeft om volle eigenaar te worden na het overlijden van de ouders. Deze techniek wordt vaak gebruikt om erfbelasting te vermijden. Het is echter belangrijk om te voldoen aan de fiscale regels en het mogelijke vermoeden van bedekte bevoordeling door de fiscus.", "userName": "Miguel"})

print(result)

