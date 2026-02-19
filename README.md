
A RAG-based code analysis tool. Upload a ZIP of any Python or JavaScript project and ask natural language questions about its architecture, logic, or individual files.

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Loader | LangChain GenericLoader + LanguageParser |
| Splitter | RecursiveCharacterTextSplitter.from_language |
| Embeddings | all-MiniLM-L6-v2 (local, CPU) |
| Vector DB | Chroma (in-memory) |
| LLM | Ollama (Llama 3) or OpenAI |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use Ollama locally, install it from https://ollama.com then run:

```bash
ollama pull llama3
ollama serve
```

To use OpenAI instead, set your key before starting the app:

```bash
export OPENAI_API_KEY="sk-..."
```

Then toggle **Use OpenAI instead of Ollama** in the sidebar.

## Features

**Chat interface** — ask questions about your codebase and get answers grounded in the actual source code.

**Source citations** — every answer includes the filename and line numbers of the retrieved chunks.

**Explain This File** — select any file from the sidebar to get a structured summary covering its purpose, key definitions, dependencies, and notable patterns.

## Supported File Types

`.py` `.js` `.ts`
