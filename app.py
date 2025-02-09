import streamlit as st
import warnings
import os
import textwrap
from dotenv import load_dotenv

# Suppress warnings if needed.
warnings.filterwarnings("ignore")

# Load environment variables. Update the .env path as needed.
load_dotenv(dotenv_path="/Users/mohsenrahimi/Documents/Final_Project/LLMenv.env")

# Import your LangChain and Neo4j-related classes.
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

# --- Helper function to initialize your RAG chain ---
def init_chain():
    # 1. Environment and Neo4j Connection
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    NEO4J_DATABASE = "finrag3"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    # 2. Initialize the LLM and Embedding Provider
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )
    embedding_provider = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    # 3. Define the Neo4j Retrieval Query
    retrieval_query = """
    MATCH window= (node:Content)-[:HAS_TEXT_CHUNK]->(chunk:TableTextChunk)-[:NEXT*0..20]-> (:TableTextChunk)
    WITH node, score, window AS longestWindow
    ORDER BY length(window) DESC LIMIT 1
    WITH nodes(longestWindow) AS chunkList, node, score
    UNWIND chunkList AS chunkRows
    WITH collect(chunkRows.text) AS textList, node, score
    RETURN apoc.text.join(textList, " \n ") AS text,
           score,
           node {.source} AS metadata
    """

    # Create the Neo4j vector index retriever.
    chunk_vector = Neo4jVector.from_existing_index(
        embedding_provider,
        graph=graph,
        index_name="content_vector_index",
        embedding_node_property="textEmbedding",
        text_node_property="text",
        retrieval_query=retrieval_query
    )

    # Create a retriever (here retrieving the top 2 documents).
    retriever = chunk_vector.as_retriever(search_kwargs={'k': 2})

    # 4. Set Up Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Build the Conversational Retrieval Chain.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return chain

# --- End of helper function ---

# To ensure that each user session gets its own chain (and its conversation history),
# we store it in Streamlit’s session state.
if "chain" not in st.session_state:
    st.session_state.chain = init_chain()

# Optionally, maintain a separate chat history list in session state to display on the page.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Each item will be a dict with 'question' and 'answer'

# --- Streamlit UI Layout ---
st.title("RAG Chatbot Demo")

# st.markdown(
#     """
#     Enter your question in the box below. The system will use your retrieval augmented generation (RAG) chain to answer your query.
#     """
# )

# Create an input form to avoid re-running the entire script on every keypress.
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your Question:", key="input")
    submit_button = st.form_submit_button(label="Send")

# When the user submits a question, get an answer from the chain.
if submit_button and user_input:
    with st.spinner("Generating answer..."):
        answer = st.session_state.chain.run(user_input)
    # Append the new Q&A to the chat history.
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": answer
    })

# --- Display the conversation history with custom font styling for the Bot's answer ---
st.markdown("### Conversation")
# Reverse the chat history so the latest question and answer appear at the top.
for chat in reversed(st.session_state.chat_history):
    # Display the user's question in bold using markdown.
    st.markdown(f"<p><strong>User:</strong> {chat['question']}</p>", unsafe_allow_html=True)

    # Wrap the bot's answer in a <p> tag with inline CSS.
    # Here we set the font to 'Courier New', font size to 16px, and the color to blue.
    st.markdown(
        f"<p style='font-family: Courier New, monospace; font-size: 16px; color: blue;'><strong>RAG:</strong> {chat['answer']}</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

