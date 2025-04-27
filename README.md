# simpleRAG

**simpleRAG** is a fully local, private, and powerful RAG (Retrieval-Augmented Generation) system built with Streamlit, LangChain, and Ollama. Upload your documents, create a vector-based knowledge base, and query it intelligently with local language models — all without sending any data to the cloud.

---

## Features

- 🌐 Local document ingestion (TXT, JSON)
- ⚖️ Knowledge base creation with automatic chunking and embeddings
- ✨ Query interface using local LLMs via Ollama
- ⚡ Fast, simple retraining
- 🧰 HuggingFace embeddings and ChromaDB vector storage
- 🛡️ Fully customizable model and chunk settings
- 🌎 100% local and private: no external API calls

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

You'll also need Ollama installed and running locally.

### 2. Launch the App
```bash
streamlit run app.py
```

### 3. Upload Files
- Upload `.txt` or `.json` documents (for JSON, it extracts the `.text` field).
- Click **Retrain Model** to build the knowledge base.

### 4. Ask Questions
- Type a question based on your uploaded documents.
- Get instant answers retrieved and generated locally.

---

## Configuration

| Setting | Description |
|:-------|:------------|
| **Ollama Host/Port** | Connect to your local Ollama server |
| **Model** | Select any available Ollama model |
| **Temperature** | Creativity of the LLM responses |
| **Max Tokens** | Maximum length of responses |
| **Top P / Top K** | Fine-tuning randomness |
| **Chunk Size / Overlap** | Control how documents are split |
| **Embedding Model** | Choose a HuggingFace model for vectorization |

Advanced settings are available via the sidebar in the app.

---

## How It Works

1. **Upload Documents**
   - Text and JSON files are parsed and loaded.

2. **Split into Chunks**
   - Documents are chunked for better semantic search.

3. **Embed and Store**
   - Chunks are embedded with HuggingFace models and stored in ChromaDB.

4. **Query Processing**
   - When you ask a question, the app retrieves the most relevant chunks.
   - These chunks are passed to the local LLM to generate a contextualized answer.

---

## Ollama Models

You can use any local model available on your Ollama instance, including:

- Llama2, Llama3
- Mistral
- Phi, Orca-mini
- Codellama
- Vicuna
- Neural-chat
- Stable-beluga
- Or any custom models you install (e.g., `huihui_ai/gemma3-abliterated:27b-q8_0`)

---

## Important Notes

- **Completely Local**: No information is sent outside your machine.
- **Persistence**: Chroma vectorstore is saved to disk (`./chroma_db`) between sessions.
- **Memory Usage**: Embeddings can use significant RAM depending on document size and chunk settings.

---

## Troubleshooting

- If you encounter a PyTorch warning, it will be caught and shown in the sidebar.
- Ensure Ollama is running and reachable via your specified host and port.
- If retraining fails, check that your uploaded files are valid text or JSON.

---

## License

Apache License 2.0

---

## Author

**88plug**

---

## Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.ai/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)
- [ChromaDB](https://docs.trychroma.com/)

---

## Roadmap Ideas (Optional Future Features)

- ✨ More advanced retrievers (BM25 hybrid, etc.)
- 🌍 Multi-language support
- 🎈 Fine-tuning local LLMs based on uploaded documents
- 🌐 Export and import full knowledge bases

---

Enjoy building your private RAG apps with **simpleRAG**! 🌱
