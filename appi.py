
import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.schema import Document
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
torch.classes.__path__ = []
from langchain_community.chat_models import ChatOllama


# Load environment variables from Hugging Face Secrets
load_dotenv()

os.environ['HUGGINGFACEHUB_API_KEY'] = os.getenv("HUGGINGFACEHUB_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Document Management and RAG-based Q&A Application"

# Streamlit Page Config
st.set_page_config(
    page_title="Document Management ",
    layout="centered"
)

st.title("📚 Document Management and RAG-based Q&A Application")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# A placeholder to store vector database (FAISS)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Initialize the local LLaMA model
llama_model = ChatOllama(model="llama3.2")


# Process the PDFs, Create/Update the Vector Store
if st.button("Process PDFs") and uploaded_files:
    all_documents = []

    for file in uploaded_files:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
            print(f"File saved at {temp_file_path}")

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pdf_docs = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in pdf_docs:
            if not doc.page_content.strip():
                print(f"Empty content in document: {doc.metadata}")
                continue
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                # Create Document object for each chunk
                all_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    # Create embeddings with Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_documents(
        documents=all_documents,
        embedding=embeddings
    )

    st.success("PDFs processed and vector store created!")

# Query 
query = st.text_input("Enter your question")

if st.button("Get Answer"):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs first.")
    else:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llama_model, chain_type="stuff", retriever=retriever, return_source_documents=False,output_key="result")


        # Retrieve documents and generate response
        relevant_docs = retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate answer using Hugging Face model
        response = qa_chain({"query": query})

        st.markdown("### Answer:")
        st.write(response["result"])

        with st.expander("Show source documents"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Source Document {i + 1}:**")
                st.write(doc.page_content)
                st.write("---")



                #in terminal -  huggingface-cli login
                #python -m streamlit run appi.py