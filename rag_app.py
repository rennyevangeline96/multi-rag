import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()


# ---------------------------------------------------------
# 2. Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# 3. Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }

        .source-box {
            background-color: #f5f5f5;
            padding: 10px 14px;
            border-radius: 8px;
            margin-top: 8px;
            border-left: 4px solid #555;
        }

        .stats-box {
            padding: 12px;
            border-radius: 8px;
            background-color: #f5f5f5;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# 4. Session state
# ---------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------------------------
# 5. Create FAISS vector store
# ---------------------------------------------------------

def create_vector_store(uploaded_files):

    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    progress = st.progress(0)
    status = st.empty()

    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):

        status.write(
            f"Reading {uploaded_file.name}..."
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        try:

            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Store the original uploaded filename
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            chunks = text_splitter.split_documents(pages)

            all_chunks.extend(chunks)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

        progress.progress((index + 1) / total_files)

    status.write("Creating embeddings and FAISS vector database...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        all_chunks,
        embeddings
    )

    progress.empty()
    status.success(
        f"Indexed {len(all_chunks)} chunks from {total_files} PDF files."
    )

    return vector_store, len(all_chunks)


# ---------------------------------------------------------
# 6. Helper functions
# ---------------------------------------------------------

def get_page_number(document):

    page = document.metadata.get("page")

    if page is None:
        return "Unknown"

    # PyPDFLoader page numbers start from 0
    return str(page + 1)


def get_source_name(document):

    return document.metadata.get(
        "source",
        "Unknown document"
    )


def get_unique_sources(documents):

    sources = []

    for document in documents:

        source = get_source_name(document)
        page = get_page_number(document)

        item = (source, page)

        if item not in sources:
            sources.append(item)

    return sources


# ---------------------------------------------------------
# 7. Ask question using RAG
# ---------------------------------------------------------

def answer_question(question):

    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    relevant_documents = retriever.invoke(question)

    if not relevant_documents:
        return (
            "I could not find relevant information in the uploaded documents.",
            []
        )

    context_parts = []

    for document in relevant_documents:

        source = get_source_name(document)
        page = get_page_number(document)

        context_parts.append(
            f"""
SOURCE FILE: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question ONLY using the information
provided in the document context below.

Important rules:

1. Do not invent information.
2. If the answer is not available in the documents,
   clearly say that the information was not found.
3. If two or more documents contain conflicting information,
   DO NOT silently choose one.
4. Clearly explain the conflicting information and identify
   which document contains each version.
5. Give a concise and useful answer.
6. Do not include source citations in the answer itself.
   Sources will be displayed separately by the application.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(prompt)

    sources = get_unique_sources(relevant_documents)

    return response.content, sources


# ---------------------------------------------------------
# 8. Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("📚 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} PDF(s) selected**"
        )

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        if st.button(
            "🔍 Build Knowledge Base",
            use_container_width=True
        ):

            with st.spinner(
                "Processing your documents..."
            ):

                try:

                    vector_store, chunk_count = (
                        create_vector_store(uploaded_files)
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.chunk_count = chunk_count
                    st.session_state.indexed_files = [
                        file.name for file in uploaded_files
                    ]

                    # Start a fresh conversation for a new index
                    st.session_state.messages = []

                except Exception as e:

                    st.error(
                        f"Error while processing PDFs: {e}"
                    )

    st.divider()

    st.subheader("📊 Index Status")

    if st.session_state.vector_store:

        st.success("Knowledge base ready")

        st.metric(
            "Indexed Chunks",
            st.session_state.chunk_count
        )

        st.write("**Indexed Files:**")

        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")

    else:

        st.info(
            "Upload PDFs and click "
            "'Build Knowledge Base'."
        )


# ---------------------------------------------------------
# 9. Main application
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Multi-PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions across multiple PDF documents '
    'with source and page references.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# 10. Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                st.markdown("**📌 Sources**")

                for source, page in sources:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        📄 <b>{source}</b> — Page {page}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ---------------------------------------------------------
# 11. Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)


if question:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload PDFs and build the knowledge base first."
        )

        st.stop()

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching the documents..."):

            try:

                answer, sources = answer_question(question)

                st.markdown(answer)

                if sources:

                    st.markdown("**📌 Sources**")

                    for source, page in sources:

                        st.markdown(
                            f"""
                            <div class="source-box">
                            📄 <b>{source}</b> — Page {page}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                error_message = f"Error: {e}"

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    }
                )