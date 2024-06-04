# Import things that are needed generically
from langchain.tools import tool

@tool
def search(response: str) -> str:
    """Search tool."""
    return "Searching..."