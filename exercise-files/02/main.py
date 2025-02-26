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

template = """Answer the question based only on the following contexts
 {Context}

Question: {Question}

"""

prompt = ChatPromptTemplate.from_template(template)

# create a vectorstore & embeddings : https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/#add-items-to-vector-store
# Creating embeddings: Converts text into a numerical format (vectors) that represents its meaning, making it easier for a machine to understand; 
# Vector store: Stores these embeddings in a database, allowing efficient searching and retrieval based on semantic similarity.

vectorstore = FAISS.from_texts(['harisson worked at kensho'], embedding=OpenAIEmbeddings())


# querying the vectorstore --> Define a query; Run similarity search: Use the vector store's method to search for relevant documents based on the query;
# Top_k parameter: Optionally limit the results to the top most relevant ones;Print results: Retrieve and display the content of the most relevant documents; The process returns document objects with content (and optional metadata) matching the query's embeddings.
query = "where did harrisson work?"
docs = vectorstore.similarity_search(query, top_k=1)
print(docs[0].page_content)


# querying vector store as retriever
retriever = vectorstore.as_retriever()
docs = retriever.invoke(query, top_k=1)
print(docs[0].page_content)


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