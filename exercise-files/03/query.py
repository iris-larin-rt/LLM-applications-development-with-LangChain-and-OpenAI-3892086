import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
# initialize client for OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text_to_embed):
    """Get an embedding for a piece of text."""
    response = client.embeddings.create(
        model= "text-embedding-ada-002",
        input=[text_to_embed]
    )
    print(response.data[0].embedding)

template: str = """/
    You are a customer support specialist /
    question: {question}. 
    You assist users with general inquiries based on {context} /
    and  technical issues. /
    """

# define prompt
system_message_prompt_template = SystemMessagePromptTemplate.from_template(template)
human_message_prompt_template = HumanMessagePromptTemplate.from_template(input_variables=["question","context"], template="{question}")

chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, human_message_prompt_template])



# init model
model = ChatOpenAI()



# Step 1: indexing --> load document and split into chunks --> indexing using  langchain_community.document_loaders import TextLoader
def load_split_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_text = TextLoader("./docs/faq.txt").load()
    #chunk_size=fixed number of characters(100 character per chunk), 
    #chunk_overlap=number of character to over overlap each other(what does this mean in the result)
    text_splitter = CharacterTextSplitter( chunk_size=30, chunk_overlap=0, separator=".") 
    chunks = text_splitter.split_documents(raw_text)
    # print(raw_text)
    # print(f"Number of chunks{len(chunks)}")
    # print(chunks[0])
    return chunks


# Step 2: convert chunks to embeddings (vectors) --> using chroma vector store --> langchain_community.embeddings import OpenAIEmbeddings
def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embeddings)
    docs =db.similarity_search(user_query)
    # print(docs)
    # get_embedding(user_query)
    # _ = [get_embedding(doc.page_content) for doc in docs]
    return db.as_retriever()
   

def generate_response(retriever, query):
    """Generate a response to a user query."""
    chain = (
        {"context": retriever,
        "question": RunnablePassthrough()}
        | chat_prompt_template 
        | model 
        | StrOutputParser()
        )
    return chain.invoke( query)
   

def query(query):
    """Query the model with a user query."""
    documents = load_split_documents()
    retriever= load_embeddings(documents, query)
    return generate_response(retriever, query)
    


# query("what is the return policy?")