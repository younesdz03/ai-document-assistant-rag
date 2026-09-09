import streamlit as st

from core.pdf_loader import load_pdf
from core.text_splitter import create_chunks
from core.rag_pipeline import RAGPipeline
from core.llm import generate_summary


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "current_files" not in st.session_state:
    st.session_state.current_files = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""


rag_pipeline = st.session_state.rag_pipeline


# =========================================================
# HEADER
# =========================================================

st.title("📚 AI Document Assistant")

st.write(
    "Upload one or multiple PDF documents and ask questions "
    "about their content using Gemini, RAG and ChromaDB."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Choose one or multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if uploaded_files:

    current_files = tuple(
        (file.name, file.size)
        for file in uploaded_files
    )

    if (
        not st.session_state.documents_processed
        or st.session_state.current_files != current_files
    ):

        try:

            with st.spinner("Processing documents..."):

                rag_pipeline.clear_index()

                all_pages = []

                # -------------------------------------------------
                # LOAD ALL PDF FILES
                # -------------------------------------------------

                for uploaded_file in uploaded_files:

                    pages = load_pdf(uploaded_file)

                    if not pages:
                        st.warning(
                            f"No readable text found in "
                            f"{uploaded_file.name}."
                        )
                        continue

                    all_pages.extend(pages)

                if not all_pages:

                    st.error(
                        "No readable text was found in the uploaded PDFs."
                    )

                    st.stop()

                # -------------------------------------------------
                # SAVE FULL DOCUMENT TEXT FOR SUMMARY
                # -------------------------------------------------

                full_text_parts = []

                for page in all_pages:

                    filename = page.get(
                        "filename",
                        "document.pdf"
                    )

                    page_number = page.get(
                        "page",
                        "?"
                    )

                    text = page.get(
                        "text",
                        ""
                    )

                    if text.strip():

                        full_text_parts.append(
                            f"""
Document: {filename}
Page: {page_number}

{text}
""".strip()
                        )

                st.session_state.document_text = (
                    "\n\n---\n\n".join(full_text_parts)
                )

                # -------------------------------------------------
                # CREATE CHUNKS
                # -------------------------------------------------

                all_chunks = create_chunks(
                    pages=all_pages,
                    chunk_size=1000,
                    chunk_overlap=200
                )

                if not all_chunks:

                    st.error(
                        "No chunks could be created from the documents."
                    )

                    st.stop()

                # -------------------------------------------------
                # INDEX INTO CHROMADB
                # -------------------------------------------------

                rag_pipeline.index_chunks(
                    all_chunks
                )

                st.session_state.documents_processed = True
                st.session_state.current_files = current_files

            st.success(
                f"{len(uploaded_files)} document(s) processed successfully."
            )

        except Exception as error:

            st.error(
                f"Error while processing documents: {error}"
            )

            st.stop()


    # =====================================================
    # DOCUMENT INFORMATION
    # =====================================================

    st.subheader("📄 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documents",
            len(uploaded_files)
        )

    with col2:

        st.metric(
            "Indexed chunks",
            rag_pipeline.count_documents()
        )

    with col3:

        total_size_mb = sum(
            file.size
            for file in uploaded_files
        ) / (1024 * 1024)

        st.metric(
            "Total size",
            f"{total_size_mb:.2f} MB"
        )


    # =====================================================
    # DISPLAY FILE LIST
    # =====================================================

    with st.expander("📁 Uploaded documents"):

        for index, file in enumerate(
            uploaded_files,
            start=1
        ):

            st.write(
                f"{index}. {file.name}"
            )


    # =====================================================
    # DOCUMENT SUMMARY
    # =====================================================

    st.subheader("📝 Document Summary")

    st.write(
        "Generate a structured summary of the uploaded document(s)."
    )

    if st.button(
        "Summarize documents"
    ):

        if not st.session_state.document_text.strip():

            st.warning(
                "No document text is available for summarization."
            )

        else:

            try:

                with st.spinner(
                    "Generating document summary..."
                ):

                    summary = generate_summary(
                        text=st.session_state.document_text,
                        language="French"
                    )

                st.subheader("Summary")

                st.write(summary)

            except Exception as error:

                st.error(
                    f"Error while generating summary: {error}"
                )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    st.subheader("💬 Ask your documents")

    question = st.text_input(
        "Ask a question about the uploaded documents:",
        placeholder="Example: What is the main topic discussed?"
    )


    if st.button(
        "Ask AI",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching documents and generating an answer..."
                ):

                    result = rag_pipeline.ask(
                        question=question,
                        n_results=3
                    )

                # -------------------------------------------------
                # ANSWER
                # -------------------------------------------------

                st.subheader("Answer")

                answer = result.get(
                    "answer",
                    ""
                )

                st.write(answer)


                # -------------------------------------------------
                # SOURCES
                # -------------------------------------------------

                sources = result.get(
                    "sources",
                    []
                )

                no_information_message = (
                    "Je ne trouve pas cette information dans le document."
                )

                if (
                    sources
                    and answer.strip() != no_information_message
                ):

                    st.subheader("Sources")

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        filename = source.get(
                            "filename",
                            "Unknown document"
                        )

                        page = source.get(
                            "page",
                            "?"
                        )

                        chunk_id = source.get(
                            "chunk_id",
                            ""
                        )

                        distance = source.get(
                            "distance"
                        )

                        with st.expander(
                            f"Source {index} — "
                            f"{filename} "
                            f"(Page {page})"
                        ):

                            st.write(
                                f"Chunk ID: {chunk_id}"
                            )

                            if distance is not None:

                                st.write(
                                    "Vector distance:",
                                    round(
                                        distance,
                                        4
                                    )
                                )

            except Exception as error:

                st.error(
                    f"Error while generating the answer: {error}"
                )


# =========================================================
# NO DOCUMENT
# =========================================================

else:

    st.info(
        "Please upload one or multiple PDF documents to begin."
    )