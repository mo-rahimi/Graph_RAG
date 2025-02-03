import streamlit as st

st.write("This is the project")

import streamlit as st
import os
import warnings
import textwrap
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(dotenv_path="/Users/mohsenrahimi/Documents/Final_Project/LLMenv.env")

# Fetch credentials
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = "finrag3"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up Neo4j connection
kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

# Vector store setup
neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    graph=kg,
    index_name='content_vector_index',
    node_label='Content',
    text_node_properties=['text'],
    embedding_node_property='textEmbedding'
)

# Define retriever and LLM
retriever = neo4j_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# Define chain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Streamlit UI
st.title("Neo4j-Powered Q&A Demo")
st.write("Ask any question and get responses from the Neo4j knowledge base.")

# User input
user_question = st.text_input("Enter your question:")

# Process input when button is clicked
if st.button("Get Answer"):
    if user_question:
        response = chain.invoke({"question": user_question})
        answer = textwrap.fill(response.get('answer', 'No answer found.'), width=80)
        st.success("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question.")
