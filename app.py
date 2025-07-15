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
from gtts import gTTS
from io import BytesIO

# ------------------ API Key ------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# ------------------ Page Config ------------------
st.set_page_config(page_title="📚 AI Document Assistant", layout="wide")

# ------------------ Utilities ------------------

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
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY),
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
    path = "/tmp/chat_history.pdf"
    pdf.output(path)
    return path

def generate_summary(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a document summarizer. Given the following context from a document, summarize it concisely.

Context:
{context}

Question: Please summarize the document.

Answer:"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": summary_prompt}
    )

    result = qa.run("Please summarize the document.")
    return result

def translate_text(text, lang_code):
    translation_prompt = f"Translate the following text into {lang_code}:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": translation_prompt}]
    )
    return response['choices'][0]['message']['content']

def text_to_audio(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    return mp3_fp

# ------------------ Sidebar ------------------

st.sidebar.title("📂 Document Options")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
top_k = st.sidebar.slider("Top-K Chunks to Retrieve", 1, 10, 3)
if st.sidebar.button("🧹 Clear All"):
    st.session_state.clear()
    st.rerun()

# ------------------ Session State ------------------

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'summary' not in st.session_state:
    st.session_state.summary = ""

if 'translated_summary' not in st.session_state:
    st.session_state.translated_summary = ""

# ------------------ Main App ------------------

st.title("📚 AI Document Assistant")
st.markdown("Ask questions, summarize content, and translate documents using **OpenAI + LangChain**.")

if uploaded_files:
    docs = load_documents(uploaded_files)
    chunks = split_docs(docs)
    qa_chain = init_chatbot(chunks, top_k)

    st.subheader("📝 Ask a Question")
    user_question = st.text_input("💬 Ask something from the uploaded documents:")
    if user_question:
        result = qa_chain({"query": user_question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

        st.session_state.chat_history.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "question": user_question,
            "answer": answer,
            "sources": sources
        })

        st.success("✅ Answer Generated:")
        st.markdown(answer)
        st.caption("📄 Sources: " + ", ".join(set(sources)))

        # Audio
        audio_data = text_to_audio(answer)
        st.audio(audio_data, format="audio/mp3")

    # ------------- Summarize All -------------
    st.subheader("📄 Generate Document Summary")
    if st.button("🧾 Summarize All Documents"):
        with st.spinner("Summarizing..."):
            summary = generate_summary(chunks)
            st.session_state.summary = summary
        st.success("📋 Summary Ready!")

    if st.session_state.summary:
        with st.expander("📖 Document Summary", expanded=True):
            st.markdown(st.session_state.summary)

            # Translate summary
            lang = st.selectbox("🌐 Translate Summary to Language", ["None", "French", "German", "Urdu", "Spanish", "Arabic"])
            if lang != "None":
                lang_code_map = {
                    "French": "fr", "German": "de", "Urdu": "ur",
                    "Spanish": "es", "Arabic": "ar"
                }
                translated = translate_text(st.session_state.summary, lang)
                st.session_state.translated_summary = translated
                st.markdown(f"**📘 Translated Summary ({lang}):**")
                st.markdown(translated)

                audio = text_to_audio(translated, lang=lang_code_map[lang])
                st.audio(audio, format="audio/mp3")

    if st.session_state.summary:
        st.download_button("⬇️ Download Summary (TXT)", st.session_state.summary, file_name="summary.txt")

# ------------------ Chat History ------------------

st.subheader("🧠 Chat History")
if st.session_state.chat_history:
    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"""
        <div style="background:#f1f1f1;border-radius:10px;padding:10px;margin-bottom:10px">
            <b>🕒 {chat['time']} — You:</b><br>{chat['question']}
        </div>
        <div style="background:#e0f7fa;border-radius:10px;padding:10px;margin-bottom:20px">
            <b>🤖 Answer:</b><br>{chat['answer']}<br>
            <small>📄 Sources: {', '.join(set(chat['sources']))}</small>
        </div>
        """, unsafe_allow_html=True)

# ------------------ Export Chat ------------------

if st.session_state.chat_history:
    st.subheader("📥 Export Chat History")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download TXT"):
            txt = "\n".join([f"[{c['time']}] Q: {c['question']}\nA: {c['answer']}\n" for c in st.session_state.chat_history])
            b64 = base64.b64encode(txt.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">📥 Download TXT</a>'
            st.markdown(href, unsafe_allow_html=True)

    with col2:
        if st.button("Download PDF"):
            txt = "\n".join([f"[{c['time']}] Q: {c['question']}\nA: {c['answer']}\n" for c in st.session_state.chat_history])
            path = export_pdf(txt)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="chat_history.pdf">📄 Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
