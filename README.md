# Introduction 
This Repo contains multiple PoC projects for AI using Azure OpenAI and langchain

# Getting Started
Install Python 3.11 (make sure to also install pip)
Install Poetry
In root of the project run poetry install
Then poetry shell
in VS Code -> on a python file -> in bottom right -> make sure you select the venv created by poetry as language mode

# Running Projects
Chatbot: streamlit run ./app/chat/chat_with_history.py
Index loading for coman: python ./app/test-import-coman.py
API with swagger: fastapi dev ./app/api/main.py
In discovery are all the test projects a jupyter notebook (make sure to select the venv as python interpreter when running)

# Generate requirements from poetry 
poetry export --without-hashes --format=requirements.txt > requirements.txt
