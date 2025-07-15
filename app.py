import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import openai
import tempfile
import os
import base64

import datetime
from fpdf import FPDF

# 🔐 Set your OpenAI API Key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="📚 PDF Chatbot (Memory + LangChain)", layout="wide")

# ----------- Utility Functions ----------------

def load_documents(files):
    docs = []
    for file in files:
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(documents)

def init_chatbot(documents, k):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # ✅ Define proper PromptTemplate
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Only answer using the content of the document.
If the answer isn't in the document, reply: "I'm not sure based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        ),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return qa

def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    path = "/content/chat_history.pdf"
    pdf.output(path)
    return path

# ----------- UI Setup ----------------

st.title("📚 Chat with Your Documents")
st.markdown("Upload your PDF or DOCX and ask questions. Responses are powered by **OpenAI + LangChain**.")

with st.sidebar:
    uploaded_files = st.file_uploader("📂 Upload PDFs/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, 3)
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ----------- Chat Memory Init ----------------

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    raw_docs = load_documents(uploaded_files)
    chunks = split_docs(raw_docs)
    qa_chain = init_chatbot(chunks, top_k)

    question = st.text_input("💬 Ask a question from your documents")

    if question:
        response = qa_chain({"query": question})
        answer = response["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]

        # Save to memory
        st.session_state.chat_history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": sources
        })

# ----------- Chat UI ----------------

st.subheader("🧠 Chat History")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history[::-1]:  # reverse order
        with st.container():
            st.markdown(f"""
                <div style="background:#f1f1f1;border-radius:10px;padding:10px;margin-bottom:10px">
                    <b>🕒 {chat['time']} — You:</b><br>{chat['question']}
                </div>
                <div style="background:#e0f7fa;border-radius:10px;padding:10px;margin-bottom:20px">
                    <b>🤖 Answer:</b><br>{chat['answer']}<br>
                    <small>📄 Source(s): {', '.join(set(chat['sources']))}</small>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("No chat yet. Upload files and ask your first question!")

# ----------- Export Options ----------------

if st.session_state.chat_history:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ Download Chat as TXT"):
            txt = "\n".join([f"[{c['time']}] Q: {c['question']}\nA: {c['answer']}\n" for c in st.session_state.chat_history])
            b64 = base64.b64encode(txt.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">📥 Download TXT</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col2:
        if st.button("⬇️ Download Chat as PDF"):
            txt = "\n".join([f"[{c['time']}] Q: {c['question']}\nA: {c['answer']}\n" for c in st.session_state.chat_history])
            path = export_pdf(txt)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="chat_history.pdf">📄 Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
