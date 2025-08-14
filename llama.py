import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load PDFs from folder
docs = []
for filename in os.listdir("./pdfs"):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("./pdfs", filename))
        loaded_docs = loader.load()
        docs.extend(loaded_docs)

st.write(f"📄 Loaded **{len(docs)}** PDF documents.")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(docs)
st.write(f"✂️ Split into **{len(texts)}** chunks.")

# Show a few example chunks (collapsible)
st.markdown("### Sample document chunks:")
for i, chunk in enumerate(texts[:3]):
    with st.expander(f"Chunk {i+1}"):
        st.write(chunk.page_content[:500] + "...")

# Embed chunks
embedder = SentenceTransformer('all-MiniLM-L6-v2')
with st.spinner("🔄 Computing embeddings..."):
    embeddings = embedder.encode([t.page_content for t in texts], show_progress_bar=True)

st.write(f"🧮 Embeddings shape: {embeddings.shape}")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

def call_ollama(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"❌ Error: {result.stderr.strip()}"
    return result.stdout.strip()

# Query input
query = st.text_input("Enter your question about IT infrastructure:")

if st.button("Get Answer") and query.strip():
    # Embed query
    query_embedding = embedder.encode([query]).astype('float32')

    # Search index for nearest chunks
    D, I = index.search(query_embedding, k=3)

    st.markdown("### 🔎 Retrieval results:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        with st.expander(f"Chunk {rank} — Distance: {dist:.4f}"):
            st.write(texts[idx].page_content)

    # Construct prompt for LLM
    retrieved_chunks = [texts[i].page_content for i in I[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are an expert IT assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
    st.markdown("### 📝 Prompt sent to Ollama:")
    st.code(prompt, language="markdown")

    with st.spinner("🤖 Generating answer from Mistral..."):
        answer = call_ollama(prompt)
        st.markdown("### 💬 Answer:")
        st.write(answer)
