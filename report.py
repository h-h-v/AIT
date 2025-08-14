import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import pdfplumber
import pytesseract
from PIL import Image
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
import io

# ----------------------------
# Path to your PDF
# ----------------------------
pdf_path = "./example.pdf"  # <-- Change this

if not os.path.exists(pdf_path):
    st.error(f"📄 File not found: `{pdf_path}`")
    st.stop()

# ----------------------------
# Load PDF with OCR and extract images
# ----------------------------
@st.cache_data(show_spinner=True)
def load_pdf_with_images_and_ocr(path):
    st.info("📥 Extracting text and images from PDF (with OCR fallback)...")
    docs = []
    images = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            img = page.to_image(resolution=300).original  # PIL Image
            
            # OCR fallback if no text
            if not text or len(text.strip()) < 20:
                st.warning(f"🔍 Page {i+1} has little/no text — running OCR on image...")
                text = pytesseract.image_to_string(img)

            # Store text as LangChain Document
            docs.append(Document(page_content=text, metadata={"page": i + 1}))

            # Save image in memory
            image_buffer = io.BytesIO()
            img.save(image_buffer, format="PNG")
            image_data = image_buffer.getvalue()
            images.append({"page": i + 1, "image": image_data})

    return docs, images

docs, images = load_pdf_with_images_and_ocr(pdf_path)
st.write(f"📄 Loaded **{len(docs)}** pages with text and images.")

# ----------------------------
# Show extracted images
# ----------------------------
st.markdown("### 🖼 Extracted Page Images:")
for img_data in images[:3]:  # show only first 3 pages to avoid overload
    st.image(img_data["image"], caption=f"Page {img_data['page']}", use_column_width=True)

# ----------------------------
# Chunking
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(docs)
st.write(f"✂️ Split into **{len(texts)}** chunks.")

# Sample preview
st.markdown("### 📚 Sample Text Chunks:")
for i, chunk in enumerate(texts[:3]):
    with st.expander(f"Chunk {i+1}"):
        st.write(chunk.page_content[:500] + "...")

# ----------------------------
# Compute Embeddings
# ----------------------------
@st.cache_data(show_spinner=True)
def compute_embeddings(text_list):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder.encode(text_list, show_progress_bar=True)

text_list = [t.page_content for t in texts]

with st.spinner("🔄 Computing embeddings..."):
    embeddings = compute_embeddings(text_list)

st.write(f"🧮 Embeddings shape: {embeddings.shape}")

# ----------------------------
# Build FAISS Index
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# ----------------------------
# Ollama REST API
# ----------------------------
def call_ollama(prompt, model="mistral"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code != 200:
            return f"❌ Error: {response.text}"
        return response.json()['response'].strip()
    except Exception as e:
        return f"❌ Failed to connect to Ollama: {e}"

# ----------------------------
# User Query
# ----------------------------
query = st.text_input("Enter your question about the PDF content:")

if st.button("Get Answer") and query.strip():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedder.encode([query]).astype('float32')

    # Retrieve top-k similar chunks
    D, I = index.search(query_embedding, k=3)

    st.markdown("### 🔍 Retrieved Chunks:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        with st.expander(f"Chunk {rank} — Distance: {dist:.4f}"):
            st.write(texts[idx].page_content)

    # Build prompt
    retrieved_chunks = [texts[i].page_content for i in I[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are an expert assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    st.markdown("### 📝 Prompt sent to Ollama:")
    st.code(prompt, language="markdown")

    with st.spinner("🤖 Generating answer..."):
        answer = call_ollama(prompt)
        st.markdown("### 💬 Answer:")
        st.write(answer)
