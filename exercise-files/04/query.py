import sqlite3
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever,  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
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



# START: We'll need to update two things about our existing app:
# Prompt: Update our prompt to support historical messages as an input.
# Contextualizing questions: Add a sub-chain that takes the latest user question and reformulates it in the context of the chat history. 
#   This is needed in case the latest question references some context from past messages. 
#   For example, if a user asks a follow-up question like "Can you elaborate on the second point?", 
#   this cannot be understood without the context of the previous message. Therefore we can't effectively perform retrieval with a question like this.   


contextualize_q_system_prompt = """Given a chat history and the latest user question {input}\
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# # define prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question {input}. \
based on {context}.\
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise."""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# init model
model = ChatOpenAI()
chat_history= []


# # Step 1: indexing --> load document and split into chunks --> indexing using  langchain_community.document_loaders import TextLoader
documents = TextLoader("./docs/faq.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
splits = text_splitter.split_documents(documents)
print("splits: " + str(splits))


# # Step 2: convert chunks to embeddings (vectors) --> using chroma vector store --> langchain_community.embeddings import OpenAIEmbeddings
db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory="./chroma_db")

# db = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = db.as_retriever()  
print('retrievers: ' + str(retriever))

# Step 3: Retrieve chat history: give access to  context and query/history to the model
#   First we'll need to define a sub-chain that takes historical messages and the latest user question, and reformulates the question if it makes 
#   reference to any information in the historical information.
#   We'll use a prompt that includes a MessagesPlaceholder variable under the name "chat_history". This allows us to pass in a list of Messages to the prompt 
#   using the "chat_history" input key, and these messages will be inserted after the system message and before the human message containing the latest question.
#   Note that we leverage a helper function create_history_aware_retriever for this step, which manages the case where chat_history is empty, 
#   and otherwise applies prompt | llm | StrOutputParser() | retriever in sequence.
#   create_history_aware_retriever constructs a chain that accepts keys input and chat_history as input, and has the same output schema as a retriever.
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt)
print('history_aware_retriever: ' +  str(history_aware_retriever))

# Step 4: create a chain that retrieve and generate response:
# Here we use create_stuff_documents_chain to generate a question_answer_chain, with input keys context, chat_history, and input-- 
# it accepts the retrieved context alongside the conversation history and query to generate an answer.

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
print('question_answer_chain: ' + str(question_answer_chain))

# We build our final rag_chain with create_retrieval_chain. This chain applies the history_aware_retriever and question_answer_chain 
# in sequence, retaining intermediate outputs such as the retrieved context for convenience. It has input keys input and chat_history, 
# and includes input, chat_history, context, and answer in its output.
def generate_response(query):
    """ Generate a response to a user query"""
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })

def query(query):
    """Query the model with a user query."""
    response = generate_response(query)
    chat_history.extend([HumanMessage(content=query), response["answer"]])
    return response


# query("what is the return policy?")