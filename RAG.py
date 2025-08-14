import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MyRAGBot/1.0"
import subprocess
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set USER_AGENT env variable to avoid warnings
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MyRAGBot/1.0"
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Ensure USER_AGENT is set early
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MyRAGBot/1.0"

# List of tech document URLs (PDFs or web pages)
# List of URLs to load documents from (fixed malformed URL)
urls = [
    "https://i.dell.com/sites/csdocuments/Product_Docs/en/Dell-EMC-PowerEdge-Rack-Servers-Quick-Reference-Guide.pdf",
    "https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-r660xs-technical-guide.pdf",
    "https://i.dell.com/sites/csdocuments/shared-content_data-sheets_documents/en/aa/poweredge_r740_r740xd_technical_guide.pdf",
    "https://dl.dell.com/topicspdf/openmanage-server-administrator-v95_users-guide_en-us.pdf",
    "https://dl.dell.com/manuals/common/dellemc-server-config-profile-refguide.pdf",
    "https://www.redbooks.ibm.com/redbooks/pdfs/sg248513.pdf",
    "https://www.ibm.com/docs/SSLVMB_28.0.0/pdf/IBM_SPSS_Statistics_Server_Administrator_Guide.pdf",
    "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
    "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-file",  # fixed: removed trailing >
    "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
    "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
    "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
    "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",
    "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
    "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
    "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",
    "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
    "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
    "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
    "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf",
    "https://www.dell.com/en-us/lp/dt/end-user-computing",
    "https://www.nutanix.com/solutions/end-user-computing",
    "https://eucscore.com/docs/tools.html",
    "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
]
# Load documents from URLs
docs = []
with st.spinner("Loading and parsing documents..."):
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.split_documents(docs)

# Embed using SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode([t.page_content for t in texts], show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Ollama call function (local LLM like mistral)
def call_ollama(prompt, model="mistral"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    return result.stdout.strip()

# Streamlit UI
st.title("IT Infrastructure RAG Assistant")
query = st.text_input("Enter your question about IT infrastructure:")

if st.button("Get Answer") and query.strip() != "":
    with st.spinner("Finding relevant info..."):
        query_embedding = embedder.encode([query]).astype('float32')
        D, I = index.search(query_embedding, k=3)  # Get top 3 chunks
        retrieved_chunks = [texts[i].page_content for i in I[0]]
        context = "\n\n---\n\n".join(retrieved_chunks)

        prompt = f"""You are an expert IT assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
    with st.spinner("Generating answer using Mistral..."):
        answer = call_ollama(prompt)
        st.markdown("### Answer:")
        st.write(answer)
