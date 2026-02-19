"""app.py – Streamlit frontend for Repo-Mind."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

import rag_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Repo-Mind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stChatMessage"] { max-width: 92%; }
        [data-testid="stExpander"] summary { font-size: 0.82rem; color: #888; }
    </style>
    """,
    unsafe_allow_html=True,
)

_SESSION_DEFAULTS: dict[str, Any] = {
    "vector_store": None,
    "file_paths": [],
    "repo_dir": None,
    "qa_chain": None,
    "llm": None,
    "chat_history": [],
    "selected_file": None,
    "use_openai": False,
    "openai_key": "",
    "ollama_model": rag_engine.OLLAMA_MODEL,
}


def _init_session_state() -> None:
    for key, value in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_source_citations(source_docs: list[Any]) -> str:
    seen: set[str] = set()
    lines: list[str] = []

    for doc in source_docs:
        meta = doc.metadata
        rel_src: str = (
            meta.get("relative_source") or Path(meta.get("source", "unknown")).name
        )
        citation = f"`{rel_src}`, lines {meta.get('start_line', '?')}–{meta.get('end_line', '?')}"
        if citation not in seen:
            seen.add(citation)
            lines.append(f"- {citation}")

    return "\n".join(lines)


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📎 Source Citations", expanded=False):
                    st.markdown(message["citations"])


def _process_upload(uploaded_file: Any) -> None:
    with st.spinner("⏳ Processing repository… extracting, parsing, and embedding."):
        tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip", prefix="repomind_upload_")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(uploaded_file.read())

            try:
                vector_store, file_paths, repo_dir = rag_engine.ingest_repository(tmp_zip_path)
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")
                logger.exception("Ingestion error.")
                return
        finally:
            if Path(tmp_zip_path).exists():
                os.unlink(tmp_zip_path)

        try:
            llm = rag_engine.get_llm(
                force_openai=st.session_state.use_openai,
                openai_api_key=st.session_state.openai_key or None,
                ollama_model=st.session_state.ollama_model,
            )
            qa_chain = rag_engine.build_qa_chain(vector_store, llm)
        except Exception as exc:
            st.error(f"❌ LLM initialisation failed: {exc}")
            rag_engine.cleanup_temp_dir(repo_dir)
            return

        st.session_state.vector_store = vector_store
        st.session_state.file_paths = file_paths
        st.session_state.repo_dir = repo_dir
        st.session_state.qa_chain = qa_chain
        st.session_state.llm = llm
        st.session_state.chat_history = []

    st.success(f"✅ Repository indexed – {len(file_paths)} source file(s) ready.")
    st.rerun()


def _run_file_explanation(relative_path: str) -> None:
    if not st.session_state.llm or not st.session_state.repo_dir:
        st.warning("⚠️ Please process a repository first.")
        return

    with st.spinner(f"🔍 Analysing `{relative_path}`…"):
        try:
            summary = rag_engine.explain_file(
                relative_path=relative_path,
                repo_dir=st.session_state.repo_dir,
                llm=st.session_state.llm,
            )
        except Exception as exc:
            st.error(f"❌ Could not explain file: {exc}")
            logger.exception("File explanation error.")
            return

    st.session_state.chat_history.append(
        {"role": "user", "content": f"✨ **Explain This File:** `{relative_path}`"}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": summary, "citations": ""}
    )
    st.rerun()


def _reset_session() -> None:
    if st.session_state.repo_dir:
        rag_engine.cleanup_temp_dir(st.session_state.repo_dir)

    for key, value in {
        "vector_store": None,
        "file_paths": [],
        "repo_dir": None,
        "qa_chain": None,
        "llm": None,
        "chat_history": [],
        "selected_file": None,
    }.items():
        st.session_state[key] = value

    st.rerun()


def _handle_chat_input(user_input: str) -> None:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "citations": ""}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching codebase…"):
            try:
                result: dict[str, Any] = st.session_state.qa_chain.invoke(
                    {"query": user_input}
                )
                answer: str = result.get("result", "No answer was generated.")
                citations: str = _format_source_citations(result.get("source_documents", []))
            except Exception as exc:
                answer = f"❌ An error occurred: {exc}"
                citations = ""
                logger.exception("QA chain invocation error.")

        st.markdown(answer)
        if citations:
            with st.expander("📎 Source Citations", expanded=True):
                st.markdown(citations)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 Repo-Mind")
        st.caption("RAG-based Code Analysis Tool")
        st.divider()

        st.markdown("#### 📁 Upload Repository")
        uploaded_file = st.file_uploader(
            label="Upload a ZIP of your project",
            type=["zip"],
            help="Supported: Python (.py), JavaScript (.js, .ts).",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.session_state.vector_store is not None:
                st.info("A repository is already loaded. Reset the session below to upload a new one.")
            else:
                if st.button("🚀 Process Repository", use_container_width=True):
                    _process_upload(uploaded_file)

        if st.session_state.vector_store is not None:
            st.success("✅ Repository loaded")
            st.caption(f"{len(st.session_state.file_paths)} source file(s) indexed.")

        st.divider()

        st.markdown("#### ⚙️ Model Settings")
        use_openai: bool = st.toggle(
            "Use OpenAI instead of Ollama",
            value=st.session_state.use_openai,
            help="OFF = local Ollama server. ON = OpenAI API.",
        )
        st.session_state.use_openai = use_openai

        if use_openai:
            st.session_state.openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_key,
                type="password",
                placeholder="sk-…",
                help="Leave blank to use the OPENAI_API_KEY environment variable.",
            )
        else:
            st.session_state.ollama_model = st.text_input(
                "Ollama Model",
                value=st.session_state.ollama_model,
                placeholder="llama3",
                help="Model tag on your local Ollama server.",
            )

        st.divider()

        if st.session_state.file_paths:
            st.markdown("#### 🔍 Explain This File")
            selected: str = st.selectbox(
                label="Select a file",
                options=st.session_state.file_paths,
                index=0,
                label_visibility="collapsed",
            )
            st.session_state.selected_file = selected
            display_name = selected if len(selected) <= 45 else f"…{selected[-42:]}"
            st.caption(f"📄 `{display_name}`")

            if st.button("✨ Explain File", use_container_width=True):
                _run_file_explanation(selected)

            st.divider()

        if st.session_state.vector_store is not None:
            if st.button(
                "🗑️ Reset Session",
                use_container_width=True,
                type="secondary",
                help="Clear the repository index and chat history.",
            ):
                _reset_session()


def _render_landing() -> None:
    st.markdown("### Get started in three steps")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "**📁 Step 1 – Upload**\n\n"
            "Upload a ZIP archive of any Python or JavaScript project from the sidebar."
        )
    with col2:
        st.info(
            "**⚙️ Step 2 – Configure**\n\n"
            "Choose between a local Ollama (Llama 3) instance or the OpenAI API."
        )
    with col3:
        st.info(
            "**💬 Step 3 – Ask**\n\n"
            "Ask questions about architecture, logic, or specific files."
        )

    st.markdown("---")
    st.markdown(
        "**Supported:** `.py` · `.js` · `.ts` — "
        "**Embeddings:** `all-MiniLM-L6-v2` — "
        "**Vector DB:** Chroma (in-memory) — "
        "**Retrieval:** MMR top-6"
    )


def main() -> None:
    _init_session_state()
    _render_sidebar()

    header_col, badge_col = st.columns([6, 1])
    with header_col:
        st.title("🧠 Repo-Mind")
        st.caption("Ask questions about your codebase. Source citations are provided for every answer.")
    with badge_col:
        if st.session_state.vector_store is not None:
            st.success("Indexed", icon="✅")

    st.divider()

    if st.session_state.vector_store is None:
        _render_landing()
        return

    _render_chat_history()

    user_input: str | None = st.chat_input(
        placeholder="Ask about your codebase… e.g. 'How does authentication work?'",
        disabled=st.session_state.qa_chain is None,
    )
    if user_input:
        _handle_chat_input(user_input)


if __name__ == "__main__":
    main()
