import os
import re
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv

# =========================
# Environment & LLM Setup
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# =========================
# Paths / Constants
# =========================
VECTOR_STORE_DIR = "vector_store"
DEFAULT_E2E_DIR = "cypress/e2e"

# =========================
# Workflow State
# =========================
class QAState(dict):
    requirements: List[str] = []
    out_dir: str = DEFAULT_E2E_DIR
    run_cypress: bool = False
    docs_dir: Optional[str] = None
    persist_vector_store: bool = False
    generated_files: List[str] = []
    errors: List[str] = []

# =========================
# Utilities
# =========================
def slugify(text: str, fallback: str = "spec") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text or fallback

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================
# CLI Parsing
# =========================
def parse_cli_args(state: QAState) -> QAState:
    parser = argparse.ArgumentParser(
        description="Generate Cypress tests from natural-language requirements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "requirements",
        nargs="+",
        help="One or more requirements. Each becomes its own Cypress spec."
    )
    parser.add_argument("--out", dest="out_dir", default=DEFAULT_E2E_DIR,
                        help="Output directory for generated Cypress specs.")
    parser.add_argument("--run", dest="run_cypress", action="store_true",
                        help="Run Cypress after generating tests.")
    parser.add_argument("--docs", dest="docs_dir", default=None,
                        help="Optional directory of additional context files to index.")
    parser.add_argument("--persist-vstore", dest="persist_vector_store", action="store_true",
                        help="Create/update Chroma vector store from --docs (if provided).")

    args = parser.parse_args()
    state["requirements"] = args.requirements
    state["out_dir"] = args.out_dir
    state["run_cypress"] = args.run_cypress   # <-- fixed here
    state["docs_dir"] = args.docs_dir
    state["persist_vector_store"] = args.persist_vector_store
    return state

# =========================
# Vector Store Handling
# =========================
def create_or_update_vector_store(state: QAState) -> QAState:
    docs_dir = state.get("docs_dir")
    persist = state.get("persist_vector_store", False)
    if not (docs_dir and persist):
        return state

    if not os.path.isdir(docs_dir):
        print(f"⚠️  --docs directory not found: {docs_dir}. Skipping vector store.")
        return state

    print(f"📚 Indexing documents from: {docs_dir}")
    loader = DirectoryLoader(docs_dir, glob="**/*.*")
    documents = loader.load()

    if not documents:
        print("⚠️  No documents found to index. Skipping vector store.")
        return state

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        db.add_documents(chunks)
        db.persist()
    else:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_DIR)
        db.persist()

    print("✅ Vector store updated.")
    return state

# =========================
# Test Generation
# =========================
CY_PROMPT_TEMPLATE = """You are a senior automation engineer.
Write a Cypress test in JavaScript for the following requirement:

Requirement:
{requirement}

Constraints:
- Use Cypress best practices.
- Use `describe` and `it` blocks.
- Prefer **real, working selectors from the page** (id, class, name) over non-existent data-testid.
- Include clear assertions for both success and error messages.
- Handle forms, buttons, and navigation.
- Use `cy.visit('https://the-internet.herokuapp.com/login')` as the base URL.
- Include both a positive and a negative path when applicable.
- Do not include explanations or markdown; return ONLY runnable JavaScript code.
- Ensure the code is ready to run in a standard Cypress setup.
"""

def generate_cypress_test(requirement: str) -> str:
    prompt = CY_PROMPT_TEMPLATE.format(requirement=requirement)
    result = llm.invoke(prompt)
    code = (getattr(result, "content", None) or str(result)).strip()
    return code

def generate_tests(state: QAState) -> QAState:
    out_dir = state["out_dir"]
    ensure_dir(out_dir)

    generated_files: List[str] = []
    for idx, req in enumerate(state["requirements"], start=1):
        try:
            print(f"\n[>] Generating for requirement {idx}: {req}")
            code = generate_cypress_test(req)
            slug = slugify(req)[:60]
            filename = f"{idx:02d}_{slug}_{now_stamp()}.cy.js"
            filepath = str(Path(out_dir) / filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"// Requirement: {req}\n")
                f.write(code if code.endswith("\n") else code + "\n")
            print(f"✅ Saved: {filepath}")
            generated_files.append(filepath)
        except Exception as e:
            err = f"Failed to generate for requirement {idx}: {e}"
            print(f"❌ {err}")
            state.setdefault("errors", []).append(err)

    state["generated_files"] = generated_files
    return state

# =========================
# Cypress Runner
# =========================
def run_cypress(state: QAState) -> QAState:
    if not state.get("run_cypress"):
        return state

    specs = state.get("generated_files", [])
    if not specs:
        print("⚠️  No generated specs to run.")
        return state

    print("\n▶️  Running Cypress on generated specs...")
    try:
        spec_arg = ",".join(specs)
        subprocess.run(
            ["npx", "cypress", "run", "--spec", spec_arg],
            shell=True,
            check=True
        )
        print("✅ Cypress run completed successfully.")
    except subprocess.CalledProcessError as e:
        msg = f"Error running Cypress: {e}"
        print(f"❌ {msg}")
        state.setdefault("errors", []).append(msg)
    return state

# =========================
# LangGraph Workflow
# =========================
def create_workflow():
    graph = StateGraph(QAState)
    graph.add_node("ParseCLI", parse_cli_args)
    graph.add_node("BuildVectorStore", create_or_update_vector_store)
    graph.add_node("GenerateTests", generate_tests)
    graph.add_node("RunCypress", run_cypress)

    graph.set_entry_point("ParseCLI")
    graph.add_edge("ParseCLI", "BuildVectorStore")
    graph.add_edge("BuildVectorStore", "GenerateTests")
    graph.add_edge("GenerateTests", "RunCypress")
    graph.add_edge("RunCypress", END)
    return graph.compile()

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("🚀 Starting CLI-driven QA Test Generation Workflow")
    try:
        app = create_workflow()
        state = QAState()

        # Use .invoke() instead of .run()
        state = app.invoke(state)

        print("\n✅ Done.")
        if state.get("errors"):
            print("⚠️ Completed with warnings/errors:")
            for e in state["errors"]:
                print(f" - {e}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
