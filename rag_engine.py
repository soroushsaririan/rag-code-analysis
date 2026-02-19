
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL: str = "llama3"
OPENAI_MODEL: str = "gpt-3.5-turbo"
CHUNK_SIZE: int = 2000
CHUNK_OVERLAP: int = 200
RETRIEVER_K: int = 6
OLLAMA_BASE_URL: str = "http://localhost:11434"
_EXPLAIN_FILE_MAX_CHARS: int = 8_000

_EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.JS,
}

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Repo-Mind, an expert code analysis assistant.\n"
        "Use ONLY the following retrieved code snippets to answer the question.\n"
        "After your answer, you MUST cite every source you used in this exact format:\n\n"
        "Sources:\n"
        "- <filename>, lines <start_line>-<end_line>\n\n"
        "If the retrieved context does not contain enough information to answer, "
        "state that clearly rather than guessing.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


def extract_zip(zip_path: str) -> str:
    """Extract a ZIP archive to a fresh temporary directory and return its path."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"'{zip_path}' is not a valid ZIP archive.")

    tmp_dir = tempfile.mkdtemp(prefix="repomind_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    logger.info("Extracted archive '%s' → '%s'.", zip_path, tmp_dir)
    return tmp_dir


def cleanup_temp_dir(tmp_dir: str) -> None:
    """Remove a temporary directory created by extract_zip."""
    if tmp_dir and Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Removed temporary directory '%s'.", tmp_dir)


def load_documents(repo_dir: str) -> list[Document]:
    """Load Python and JavaScript files using GenericLoader + LanguageParser."""
    all_docs: list[Document] = []

    language_globs: dict[Language, list[str]] = {}
    for ext, lang in _EXTENSION_TO_LANGUAGE.items():
        language_globs.setdefault(lang, []).append(f"**/*{ext}")

    for language, globs in language_globs.items():
        for glob_pattern in globs:
            try:
                loader = GenericLoader.from_filesystem(
                    path=repo_dir,
                    glob=glob_pattern,
                    parser=LanguageParser(language=language, parser_threshold=500),
                )
                docs = loader.load()
                all_docs.extend(docs)
                logger.debug("Loaded %d doc(s) via '%s'.", len(docs), glob_pattern)
            except Exception as exc:
                logger.warning("GenericLoader failed for '%s': %s – skipping.", glob_pattern, exc)

    logger.info("Total documents loaded before splitting: %d.", len(all_docs))
    return all_docs


def _compute_line_range(
    chunk_content: str,
    file_text: str,
    search_from: int = 0,
) -> tuple[int, int]:
    """Return 1-based (start_line, end_line) for a chunk within its source file."""
    stripped = chunk_content.strip()
    idx = file_text.find(stripped, search_from)

    if idx == -1:
        anchor = next(
            (ln.strip() for ln in chunk_content.splitlines() if ln.strip()), ""
        )
        idx = file_text.find(anchor, search_from) if anchor else -1
        if idx == -1:
            return (1, 1)

    start_line = file_text[:idx].count("\n") + 1
    end_line = start_line + chunk_content.count("\n")
    return (start_line, end_line)


def annotate_line_numbers(docs: list[Document], repo_dir: str) -> list[Document]:
    """Inject start_line, end_line, and relative_source metadata into every document."""
    file_text_cache: dict[str, str] = {}
    file_cursor: dict[str, int] = {}

    for doc in docs:
        source: str = doc.metadata.get("source", "")

        if not source or not Path(source).is_file():
            doc.metadata.setdefault("start_line", 1)
            doc.metadata.setdefault("end_line", 1)
            doc.metadata.setdefault("relative_source", source)
            continue

        if source not in file_text_cache:
            try:
                file_text_cache[source] = Path(source).read_text(
                    encoding="utf-8", errors="replace"
                )
                file_cursor[source] = 0
            except OSError as exc:
                logger.warning("Could not read '%s': %s.", source, exc)
                doc.metadata.setdefault("start_line", 1)
                doc.metadata.setdefault("end_line", 1)
                doc.metadata.setdefault("relative_source", source)
                continue

        file_text = file_text_cache[source]
        cursor = file_cursor[source]

        start_line, end_line = _compute_line_range(
            doc.page_content, file_text, search_from=cursor
        )
        doc.metadata["start_line"] = start_line
        doc.metadata["end_line"] = end_line

        match_idx = file_text.find(doc.page_content.strip(), cursor)
        if match_idx != -1:
            file_cursor[source] = match_idx + len(doc.page_content)

        try:
            rel = Path(source).relative_to(repo_dir).as_posix()
        except ValueError:
            rel = source
        doc.metadata["relative_source"] = rel

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Apply a language-aware second-pass split to chunks that exceed CHUNK_SIZE."""
    result: list[Document] = []

    for doc in docs:
        suffix = Path(doc.metadata.get("source", "")).suffix.lower()
        language = _EXTENSION_TO_LANGUAGE.get(suffix, Language.PYTHON)

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        sub_chunks: list[Document] = splitter.split_documents([doc])

        if len(sub_chunks) > 1:
            parent_start: int = doc.metadata.get("start_line", 1)
            parent_text: str = doc.page_content

            for chunk in sub_chunks:
                char_offset: int = chunk.metadata.get("start_index", 0)
                lines_before = parent_text[:char_offset].count("\n")
                chunk.metadata["start_line"] = parent_start + lines_before
                chunk.metadata["end_line"] = (
                    chunk.metadata["start_line"] + chunk.page_content.count("\n")
                )
                chunk.metadata.setdefault(
                    "relative_source", doc.metadata.get("relative_source", "")
                )
        else:
            for chunk in sub_chunks:
                chunk.metadata.setdefault("start_line", doc.metadata.get("start_line", 1))
                chunk.metadata.setdefault("end_line", doc.metadata.get("end_line", 1))
                chunk.metadata.setdefault(
                    "relative_source", doc.metadata.get("relative_source", "")
                )

        result.extend(sub_chunks)

    logger.info("Total chunks after second-pass split: %d.", len(result))
    return result


def build_vector_store(docs: list[Document]) -> Chroma:
    """Embed documents with all-MiniLM-L6-v2 and store vectors in an in-memory Chroma collection."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma.from_documents(docs, embeddings)
    logger.info("Vector store built – %d vectors indexed.", len(docs))
    return vector_store


def get_llm(
    force_openai: bool = False,
    openai_api_key: Optional[str] = None,
    ollama_base_url: str = OLLAMA_BASE_URL,
    ollama_model: str = OLLAMA_MODEL,
) -> BaseChatModel:
    """Return a chat model, preferring local Ollama; falls back to ChatOpenAI."""
    if not force_openai:
        try:
            import httpx

            response = httpx.get(f"{ollama_base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                try:
                    from langchain_ollama import ChatOllama
                except ImportError:
                    from langchain_community.chat_models import ChatOllama  # type: ignore[no-redef]

                logger.info("Ollama server reachable – using model '%s'.", ollama_model)
                return ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=0)
        except Exception as exc:
            logger.warning("Ollama health-check failed (%s) – falling back to OpenAI.", exc)

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "No LLM backend is reachable. "
            "Run Ollama locally (https://ollama.com) or set OPENAI_API_KEY."
        )

    from langchain_openai import ChatOpenAI

    logger.info("Using OpenAI model '%s'.", OPENAI_MODEL)
    return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0)


class _QAChain:
    """Minimal RAG chain using langchain-core only; returns {"result", "source_documents"}."""

    def __init__(self, retriever: object, llm: BaseChatModel, prompt: PromptTemplate) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt

    def invoke(self, inputs: dict[str, str]) -> dict[str, object]:
        """Run one QA turn. Expects {"query": str}, returns {"result", "source_documents"}."""
        query: str = inputs["query"]
        source_docs: list[Document] = self._retriever.invoke(query)
        context: str = "\n\n".join(doc.page_content for doc in source_docs)
        formatted_prompt: str = self._prompt.format(context=context, question=query)
        response = self._llm.invoke(formatted_prompt)
        answer: str = response.content if hasattr(response, "content") else str(response)
        return {"result": answer, "source_documents": source_docs}


def build_qa_chain(vector_store: Chroma, llm: BaseChatModel) -> _QAChain:
    """Build an MMR-backed QA chain from the given vector store and LLM."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3},
    )
    return _QAChain(retriever=retriever, llm=llm, prompt=_QA_PROMPT)


def explain_file(relative_path: str, repo_dir: str, llm: BaseChatModel) -> str:
    """Send the full file content to the LLM and return a structured markdown summary."""
    abs_path = Path(repo_dir) / relative_path
    if not abs_path.is_file():
        raise FileNotFoundError(
            f"Source file not found: '{abs_path}'. The repository may have been cleaned up."
        )

    source_code = abs_path.read_text(encoding="utf-8", errors="replace")
    truncated = source_code[:_EXPLAIN_FILE_MAX_CHARS]
    truncation_notice = ""
    if len(source_code) > _EXPLAIN_FILE_MAX_CHARS:
        truncation_notice = "\n\n*[File truncated – showing first 8 000 characters]*"

    prompt = (
        f"Analyse the following source file: `{relative_path}`\n\n"
        "Provide a structured summary with **exactly** these four sections:\n\n"
        "1. **Purpose** – What is the responsibility of this file within the project?\n"
        "2. **Key Classes / Functions** – List each top-level definition and describe "
        "its role in one or two sentences.\n"
        "3. **Dependencies** – What does this file import or depend on externally?\n"
        "4. **Notable Patterns** – Identify design patterns, algorithms, potential "
        "issues, or anything architecturally interesting.\n\n"
        f"```\n{truncated}\n```{truncation_notice}"
    )

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def ingest_repository(zip_path: str) -> tuple[Chroma, list[str], str]:
    """Run the full ingestion pipeline: extract → load → annotate → split → embed.

    Returns (vector_store, sorted_relative_file_paths, repo_dir).
    Caller must call cleanup_temp_dir(repo_dir) when the session ends.
    """
    repo_dir = extract_zip(zip_path)

    docs = load_documents(repo_dir)
    if not docs:
        cleanup_temp_dir(repo_dir)
        raise ValueError("No supported source files (.py, .js, .ts) found in the archive.")

    docs = annotate_line_numbers(docs, repo_dir)
    split_docs = split_documents(docs)
    vector_store = build_vector_store(split_docs)

    file_paths: list[str] = sorted(
        {doc.metadata["relative_source"] for doc in docs if doc.metadata.get("relative_source")}
    )

    return vector_store, file_paths, repo_dir
