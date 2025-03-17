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
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain import hub
from dotenv import load_dotenv

load_dotenv()


# https://python.langchain.com/docs/integrations/components/
# https://python.langchain.com/api_reference/chroma/vectorstores.html
# https://python.langchain.com/docs/how_to/query_multiple_retrievers/#create-index

# ******Language Model with Function Calling********
llm = ChatOpenAI(temperature=0)
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
    category: str = Field(
        ...,
        description="Category to look things up for. Should be `shirts` or `shoes`.",
    )

# system message -intermediary step before generating the answer
system = """You have the ability to issue search queries to get information to help answer user information based on the clothes category: 'shirts' or 'shoes'. Only questions about these categories should be answered."""

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

# ******Prompt for Retrieval and Generation Tasks
template: str = """/
            You are a customer support specialist/
            question: {question}.
            You assist user with general inquiries based on {context} and technical issues."""


 # define prompt
system_message_prompt_template = SystemMessagePromptTemplate.from_template(template)
human_message_prompt_template = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"], 
    template="{question}"
)
chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_prompt_template, 
    human_message_prompt_template
])    


# ********Create Index & Connect to datasource
# Connect to multiple datasource and create an index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

text_shoes = TextLoader("data/faq_shoes.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 0 )
documents = text_splitter.split_documents(text_shoes)
vector_store = Chroma.from_documents(documents, embeddings)
retriever_shoes = vector_store.as_retriever()

text_shirts = TextLoader("data/faq_shirts.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap = 0 )
documents = text_splitter.split_documents(text_shirts)
vector_store = Chroma.from_documents(documents, embeddings)
retriever_shirts = vector_store.as_retriever()



# ******Retrieval with Query analysis in two steps

retrievers = {
    "shoes": retriever_shoes,
    "shirts": retriever_shirts,
}

# 1.a Query Analysis: The user query is first processed by a query_analyzer, which uses a RunnablePassthrough to handle the query. 
# The goal is to direct the query towards the right category (e.g., "shirts" or "shoes") by issuing search queries through a language model.

# 1.b Category Selection: Based on the analysis, the query is categorized into either shirts or shoes. 
# This helps determine the appropriate retriever to use for retrieving relevant information.

# 1.c Structured Output: The output from the query_analyzer is structured, containing information about the query and the selected category. 
# This structured output guides the process of selecting the correct retriever for the next steps.

# Retrieving Information: After selecting the appropriate category and retriever, the system retrieves context-specific information related to the user query.
def select_retriever_query_analysis(question):
    structured_output= query_analyzer.invoke({"question": question})
    category =structured_output.category
    return retrievers[category]


#2.a Final Response Generation: The final step involves generating a natural language response by invoking a language model with the relevant context. 
# A chat_prompt_template is used to guide the language model, acting as a customer service specialist to assist with general user inquiries.

#2.b  Execution Flow: The entire process is orchestrated through a chain of components, invoking each step in sequence, 
# including passing the query, selecting the category, retrieving relevant information, and generating a response. The final output is a string-based response that is ready for the user.
def query(user_query:str):
    """Final chain to query, retrieve information and generate augment response"""
    retriever = select_retriever_query_analysis(user_query)
    return (  
        {"context": retriever,"question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
      ).invoke(user_query)
    print(retriever)



response = query("How long do I have to return a shirt")    
print(response)