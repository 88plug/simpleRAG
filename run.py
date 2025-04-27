# simpleRAG - Local Knowledge Retrieval App

import tempfile
import os
import time
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Add error handling for torch-related errors
try:
    import torch
    torch.set_num_threads(1)
except Exception as e:
    st.sidebar.warning(f"PyTorch initialization warning: {e}")

# Suppress non-critical tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Sidebar - Model Configuration
st.sidebar.header("Model Configuration")

# Ollama connection settings
ollama_host = st.sidebar.text_input("Ollama Host", value="192.168.2.11")
ollama_port = st.sidebar.number_input("Ollama Port", min_value=1, max_value=65535, value=11434)
ollama_base_url = f"http://{ollama_host}:{ollama_port}"

# Select Ollama Model
available_models = [
    "huihui_ai/gemma3-abliterated:27b-q8_0",
    "llama2", "llama3", "mistral", "phi", "orca-mini",
    "vicuna", "codellama", "neural-chat", "stable-beluga"
]
ollama_model = st.sidebar.selectbox("Select Local Ollama Model", options=available_models, index=0)

# Model settings
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 256, 4096, 2048, 256)

# Advanced Settings
with st.sidebar.expander("Advanced Settings"):
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("Top K", 1, 100, 40, 1)
    chunk_size = st.slider("Chunk Size", 1000, 4000, 2000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 400, 50)
    embedding_model = st.selectbox("Embedding Model", [
        "all-MiniLM-L6-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-mpnet-base-v2"
    ], index=0)

# Ollama LLM setup
@st.cache_resource
def get_llm(model_name, temp, base_url):
    return OllamaLLM(model=model_name, temperature=temp, base_url=base_url)

llm = get_llm(ollama_model, temperature, ollama_base_url)

@st.cache_resource
def get_embeddings(model):
    return HuggingFaceEmbeddings(model_name=model, cache_folder="./embeddings_cache", model_kwargs={'device': 'cpu'})

embeddings = get_embeddings(embedding_model)

# Main area - Title
st.markdown("""
# simpleRAG

<div class="info-box">
Local Knowledge Base Builder and Query Interface
</div>
""", unsafe_allow_html=True)

# Sidebar - System Status
st.sidebar.markdown("### System Status")
st.sidebar.markdown(f"Ollama Connection: `{ollama_base_url}`")
ollama_status = "Connected" if llm else "Not Connected"
ollama_status_color = "green" if llm else "red"
st.sidebar.markdown(f"Status: <span style='color:{ollama_status_color}'>{ollama_status}</span>", unsafe_allow_html=True)

# Test Ollama Connection
if st.sidebar.button("Test Ollama Connection"):
    try:
        import requests
        import json
        headers = {"Content-Type": "application/json"}
        data = {"model": ollama_model, "prompt": "Hello", "stream": False}
        response = requests.post(f"{ollama_base_url}/api/generate", headers=headers, data=json.dumps(data), timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ Connection successful!")
        else:
            st.sidebar.error(f"❌ Connection failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.sidebar.error(f"❌ Connection error: {str(e)}")

if "document_count" in st.session_state:
    st.sidebar.markdown(f"Chunks in memory: {st.session_state.document_count}")

# Upload Documents Section
st.header("Knowledge Base Creation")
st.markdown("Upload documents to build your searchable knowledge base.")

uploaded_files = st.file_uploader("Upload text (.txt) or JSON (.json) files", type=["txt", "json"], accept_multiple_files=True)

# Retrain Model
if st.button("Retrain Model"):
    if uploaded_files:
        with st.spinner("Processing documents and training knowledge base..."):
            try:
                documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    try:
                        if uploaded_file.name.endswith(".txt"):
                            loader = TextLoader(temp_file_path, encoding="utf-8")
                            documents.extend(loader.load())
                        elif uploaded_file.name.endswith(".json"):
                            loader = JSONLoader(file_path=temp_file_path, jq_schema='.text', text_content=False)
                            documents.extend(loader.load())
                    finally:
                        os.unlink(temp_file_path)

                if documents:
                    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    texts = text_splitter.split_documents(documents)

                    persist_directory = "./chroma_db"
                    os.makedirs(persist_directory, exist_ok=True)
                    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                    )

                    st.session_state.qa_chain = qa_chain
                    st.session_state.document_count = len(texts)
                    st.success(f"Knowledge base created with {len(texts)} text chunks!")
                else:
                    st.warning("No valid documents found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload at least one document.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.chat_history = []

# Chat Interface
st.header("Knowledge Query Interface")

chat_container = st.container()
with chat_container:
    for message in st.session_state.get("chat_history", []):
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f"**Question:** {content}")
        else:
            st.markdown(f"**Answer:** {content}")
            st.markdown("---")

if st.session_state.qa_chain:
    with st.form(key="question_form"):
        question = st.text_area("Ask a question:", height=100)
        submit_button = st.form_submit_button("Send")
        clear_button = st.form_submit_button("Clear Chat")

    if submit_button and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.qa_chain.invoke({"query": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer["result"]})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
else:
    st.info("📚 Please upload documents and retrain the knowledge base first!")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #888;">
<small>simpleRAG - Local Knowledge Retrieval - All data stays private</small>
</div>
""", unsafe_allow_html=True)
