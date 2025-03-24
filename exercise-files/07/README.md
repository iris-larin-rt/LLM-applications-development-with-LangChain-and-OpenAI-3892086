# integrate Atlas Vector Search with LangChain
> 

LLM applications with etrieval-augmented generation (RAG) to start using Atlas Vector Search with LangChain and perform semantic search 

## **Install Python** 

A [Quick Guide for Installing](https://github.com/PackeTsar/Install-Python/blob/master/README.md#install-python-) Python on Common Operating Systems

....

## Create a virtual environment :

**MacOS/Linux**:
```
python3 -m venv env
```
**Windows**:
```
python -m venv env
```

## Activate the virtual environment :
```
source env/bin/activate
```

## Installation: Install the necessary dependencies by running the following command:
**MacOS/Linux**:
```
pip3 install -r requirements.txt
pip3 install -U langchain-community
pip3 install -U langchain-openai
```
**Windows**:
```
pip install -r requirements.txt
pip install -U langchain-community
pip install -U langchain-openai
```

## [Get an API key](https://platform.openai.com/account/api-keys)

### Set the key as an environment variable:
Additionally, you need to obtain an OpenAI API key and add it to the `.env` file.

`export OPENAI_API_KEY='sk-brHeh...A39v5iXsM2'`

```
OPENAI_API_KEY=sk-brHeh...A39v5iXsM2
```

## Run the script:

**MacOS/Linux**:
```
python3 main.py
streamlit run app.py
```
**Windows**:
```
python main.py
```

## Deploy Streamlit app

1. Create an account [signup](https://share.streamlit.io/signup)
    You will be required to verify your email.
2. Create a new app - [create app](https://share.streamlit.io/new) 
3. Deploy App - [deploy](https://share.streamlit.io/deploy)