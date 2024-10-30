import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Caching the resource to avoid reloading unnecessary data
@st.cache_resource
def create_vector_index(github_url):
    # Fetch the HTML content
    response = requests.get(github_url)
    if response.status_code == 200:
        html_content = response.content
    else:
        st.error(f"Failed to retrieve content from {github_url}")
        return None
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator="\n")  # Extract all the text with line breaks
    
    # Find all hyperlinks on the page
    hyperlinks = []
    for link in soup.find_all("a", href=True):
        full_url = link['href']
        if full_url.startswith("/"):
            full_url = f"https://github.com{full_url}"  # Append the GitHub base URL if it's a relative path
        hyperlinks.append(full_url)
    
    # Add the hyperlinks to the main text content for indexing
    text += "\n" + "\n".join(hyperlinks)
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_text(text)
    documents = [Document(page_content=str(text)) for text in split_docs]

    # Create the embeddings using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Create a Chroma vector store and index
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_db")
    vectorstore_disk = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vector_index = vectorstore_disk.as_retriever(search_kwargs={"k": 5})
    
    return vector_index

# Set up the chatbot model
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)

# Question-answering chain with retrieval
def qa(vector_index):
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True)
    
    return qa_chain

def main():
    st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":brain:",
    layout="centered",
)

    # Buttons for predefined prompts
    left, middle, right = st.columns(3)

    if "vector_index" not in st.session_state:
        st.session_state.vector_index = None

    if left.button("Get started!", use_container_width=True):
        github_url = "https://github.com/isl-org/OpenBot/blob/master/README.md"
        if github_url:
            st.session_state.vector_index = create_vector_index(github_url)
            if st.session_state.vector_index:
                st.success("Vector index created! Now ask a question.")
            else:
                st.error("Failed to create vector index. Check the URL.")
    
    if middle.button("Emoji button", use_container_width=True):
        user_prompt = "This is an emoji button prompt 😊."
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

    if right.button("Material button", use_container_width=True):
        user_prompt = "This is a material button prompt 💡."
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

    # Optional input field for manual prompts
    user_input = st.chat_input("Ask Gemini-Pro...")

    if user_input:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_input)
        gemini_response = st.session_state.chat_session.send_message(user_input)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

    # Input field for the user's question
    query = st.text_input("Ask a question about the content:")

    # Check if the vector index is available for querying
    if query and st.session_state.vector_index:
        qa_chain = qa(st.session_state.vector_index)
        result = qa_chain({"query": query})
        st.write("Answer:", result["result"])

if __name__ == "__main__":
    main()
