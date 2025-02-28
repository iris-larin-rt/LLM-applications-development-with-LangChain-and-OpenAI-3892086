import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import OpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv


load_dotenv()
model = OpenAI()

template = """Answer the question based only on the following contexts and finish each sentence with an exclamation mark!
 {Context}

Question: {Question}

"""

prompt = ChatPromptTemplate.from_template(template)

# create a vectorstore & embeddings : https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/#add-items-to-vector-store
# Creating embeddings: Converts text into a numerical format (vectors) that represents its meaning, making it easier for a machine to understand; 
# Vector store: Stores these embeddings in a database, allowing efficient searching and retrieval based on semantic similarity.

vectorstore = FAISS.from_texts(['harisson worked at kensho', 'harisson is 5 years old', 'harisson like water melon','harisson like candy'], embedding=OpenAIEmbeddings())


# Example 1: Basic example: Similarity search on vector store (FAISS)
# querying the vector store --> Define a query; Run similarity search: Use the vector store's method to search for relevant documents based on the query;
# Top_k parameter: Optionally limit the results to the top most relevant ones;Print results: Retrieve and display the content of the most relevant documents; The process returns document objects with content (and optional metadata) matching the query's embeddings.
query = "what does harisson like?"
docs = vectorstore.similarity_search(query, top_k=1)
# print(docs[0].page_content)

# Example 2: Querying vector store as retriever(FAISS) using OpenAI model and Langchain to chain the retriever and model together
# querying vector store as retriever - requires model = OpenAI() where are querying vector store in similarity search does not.
retriever = vectorstore.as_retriever()
docs = retriever.invoke(query, top_k=1)
# print(docs[0].page_content)


retrieval_chain = ( 
        {
            "Context": retriever,
            "Question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
)

response = retrieval_chain.invoke(query)
print(response)