import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Configuration
st.set_page_config(
    page_title="Gemini PDF RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 Enterprise Gemini RAG Chatbot")
st.markdown("Upload a PDF document and query its contents using Google's Gemini API.")

# 1. Secure API Key Management
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.warning("⚠️ Please set your `GOOGLE_API_KEY` in Streamlit Secrets or environment variables.")
    st.stop()

# 2. Sidebar for Document Upload
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload target PDF file", type=["pdf"])
    st.markdown("---")
    st.markdown("**Engineered with:**")
    - st.markdown("- Google Gemini API")
    - st.markdown("- LangChain & ChromaDB")
    - st.markdown("- Streamlit UI")

@st.cache_resource
def load_and_vectorize_pdf(file_bytes):
    """Processes PDF bytes, chunks text, and builds an in-memory Chroma vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)  # Clean up temp file

    # Text Splitting (Chunking optimization)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Initialize Google GenAI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # Create Chroma Vector Database retriever
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Execution Pipeline Setup
if uploaded_file:
    with st.spinner("Processing PDF and generating semantic embeddings..."):
        retriever = load_and_vectorize_pdf(uploaded_file.getvalue())

    # Initialize Gemini LLM (using gemini-2.5-flash or standard production flash model)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    # Prompt Engineering for Context-Aware Retrieval
    system_prompt = (
        "You are an expert technical AI assistant. Use the provided context "
        "to answer the user's question accurately. If you don't know the answer, "
        "state clearly that the information is missing from the document.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build RAG Chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 4. Chat Interface State Management
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Query Input
    if user_query := st.chat_input("Ask a question about your PDF document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing answer with Gemini..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Please upload a PDF file via the sidebar to start querying your document.")