import streamlit as st
import tempfile
import os

from utils.rag_pipeline import process_document, ask_question

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="InterviewAce AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#1F77B4;
}

.subtitle{
    color:gray;
    font-size:18px;
    margin-bottom:10px;
}

.tech-box{
    background:#F3F6FA;
    padding:12px;
    border-radius:10px;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    '<div class="main-title">🎯 InterviewAce AI</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">Intelligent Interview Preparation Platform using Retrieval-Augmented Generation (RAG) & Large Language Models</div>',
    unsafe_allow_html=True
)

st.divider()


# -------------------------------------------------
# DASHBOARD METRICS
# -------------------------------------------------

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="🤖 LLM",
        value="Phi-3"
    )

with m2:
    st.metric(
        label="📚 Vector DB",
        value="FAISS"
    )

with m3:
    st.metric(
        label="🧠 Embedding",
        value="MiniLM"
    )

st.write("")

# -------------------------------------------------
# FEATURE CARDS
# -------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.success("📚 RAG-based Knowledge Retrieval")

with c2:
    st.success("🤖 AI Interview Assistant")

with c3:
    st.success("📄 Document Question Answering")

st.divider()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("⚙️ Interview Settings")

domain = st.sidebar.selectbox(
    "🎯 Select Domain",
    [
        "Python",
        "SQL",
        "Data Analyst",
        "AI / ML",
        "Cloud Computing",
        "HR Interview"
    ]
)

language = st.sidebar.selectbox(
    "🌍 Response Language",
    [
        "English",
        "Hindi",
        "Marathi"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "📄 Upload Interview Material",
    type=["pdf", "txt", "docx", "csv"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    with st.spinner("Processing interview documents..."):

        process_document(temp_path, domain)

    st.sidebar.success("✅ Document Processed Successfully")

st.sidebar.divider()

st.sidebar.markdown("### 🧠 Technology Stack")

st.sidebar.info("LLM : Phi-3 Mini")

st.sidebar.info("Embeddings : all-MiniLM-L6-v2")

st.sidebar.info("Vector DB : FAISS")

st.sidebar.info("Framework : LangChain")



st.sidebar.divider()

st.sidebar.markdown("## 📌 About")

st.sidebar.info(
    """
🎯 **InterviewAce AI**

An AI-powered Interview Preparation Platform built using Retrieval-Augmented Generation (RAG).

It helps students prepare for technical and HR interviews by answering questions from uploaded interview resources.

**Ideal for:**
- Students
- Freshers
- Job Seekers
- Placement Preparation
"""
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.last_docs = []
    st.session_state.last_answer = ""
    st.rerun()

# -------------------------------------------------
# CHAT
# -------------------------------------------------

st.subheader("💬 Interview Assistant")

question = st.chat_input(
    "Ask any interview question..."
)

if question:

    st.session_state.chat_history.append(
        ("You", question)
    )

    with st.spinner("InterviewAce AI is thinking..."):

        answer, docs = ask_question(question)

    st.session_state.last_docs = docs
    st.session_state.last_answer = answer

    st.session_state.chat_history.append(
        ("InterviewAce AI", answer)
    )

st.divider()

st.subheader("💡 Example Questions")

examples = [
    "Explain Python decorators.",
    "What is normalization in SQL?",
    "Tell me about yourself.",
    "Summarize this interview guide.",
    "What are the important interview topics?"
]

for q in examples:
    st.markdown(f"• {q}")
# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------

for sender, message in st.session_state.chat_history:

    if sender == "You":

        with st.chat_message("user"):

            st.write(message)

    else:

        with st.chat_message("assistant"):

            st.write(message)

# -------------------------------------------------
# DOWNLOAD RESPONSE
# -------------------------------------------------

if st.session_state.last_answer:

    st.download_button(
        label="📥 Download Last Response",
        data=st.session_state.last_answer,
        file_name="InterviewAce_Response.txt",
        mime="text/plain"
    )

# -------------------------------------------------
# SOURCE DOCUMENTS
# -------------------------------------------------

if st.session_state.last_docs:

    st.divider()

    st.subheader("📚 Retrieved Context")

    for i, doc in enumerate(st.session_state.last_docs, start=1):

        with st.expander(f"Source Chunk {i}"):

            st.write(doc.page_content)

# -------------------------------------------------
# EXAMPLE QUESTIONS
# -------------------------------------------------

st.divider()

st.subheader("💡 Try Asking")

examples = [
    "Explain Python decorators.",
    "Difference between List and Tuple.",
    "What is Normalization in SQL?",
    "Summarize this interview guide.",
    "Tell me about yourself."
]

for q in examples:
    st.markdown(f"✅ {q}")
# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown(
"""
<div class='footer'>

InterviewAce AI • Powered by LangChain • Ollama • FAISS • Streamlit

</div>
""",
unsafe_allow_html=True
)