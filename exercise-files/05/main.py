from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from dotenv import load_dotenv

load_dotenv()


# https://python.langchain.com/docs/integrations/components/
# https://python.langchain.com/api_reference/chroma/vectorstores.html
# https://python.langchain.com/docs/how_to/query_multiple_retrievers/#create-index

# ******Language Model with Function Calling********
# llm = ChatOpenAI(temperature=0)
# this is the model that will be used to call the function - model specific to function calling
llm_function_calling = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ******Prompt with Query analysis********
# Query analysis which will be in charge of handling multiple data sources and then select which data source, 
# which retriever, to use. Use function calling to give the language models the ability to make decisions based 
# on the user query and based on the context. Use function calling to structure the output which will return multiple queries.

#  class to indicate the format using pydantic- Each output will include the user's query and the targeted person
class Search(BaseModel):
    """Search for information about a person."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    person: str = Field(
        ...,
        description="Person to look things up for. Should be `HARRISON` or `ANKUSH`.",
    )

# system message -intermediary step before generating the answer
system = """You have the ability to issue search queries to get information to help answer user information."""

# promt message
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
print(prompt)

# structured  the output by using this search datatype
structured_llm = llm_function_calling.with_structured_output(Search)

# finally run chain as intermediary to generate formatted output
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

# structured_output = query_analyzer.invoke("where did Harrison Work")


# ********Create Index & Connect to datasource
# Connect to multiple datasource and create an index
embeddings = OpenAIEmbeddings()

text = ["Harrison worked at Kensho"]
vectorstore = Chroma.from_texts(text, embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

text = ["Ankush worked at facebook"]
vectorstore = Chroma.from_texts(text, embeddings, collection_name="ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})


# docs = vectorstore.similarity_search('who worked at Kensho?')
# print(docs[0].page_content)


# ******Retrieval with Query analysis


retrievers = {
    "HARRISON": retriever_harrison,
    "ANKUSH": retriever_ankush,
}

# custom chain to execute all at once, intermediary step: select the right retriever by using query analyzer.
# with Query Analyzer generate a structured output: Person and query
# select the information of the person --selects the right person/retriever based question,
# then invokes and generate the output
# uses especial decorator @chain from langchain_core.runnables import chain
@chain
def custom_chain(question):
    structured_output  = query_analyzer.invoke(question)
    retriever = retrievers[structured_output.person]
    return retriever.invoke(structured_output.query)

response = custom_chain.invoke("where did Harrison Work")    
print(response[0].page_content)