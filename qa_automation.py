#!/usr/bin/env python3
"""
AI-Powered Cypress & Playwright Test Generator with LangGraph & Vector Store
"""

import os
import re
import sys
import json
import argparse
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"
VECTOR_DB_DIR = Path(__file__).parent / "vector_db"

LLM_CONFIG = {
    "openai": {
        "name": "OpenAI (ChatGPT)",
        "model": "gpt-4o-mini",
        "provider": "openai",
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
    },
    "google": {
        "name": "Google (Gemini)",
        "model": "gemini-2.0-flash",
        "provider": "google",
    },
}

DEFAULT_LLM = "openai"

def get_llm(provider: str = DEFAULT_LLM) -> Any:
    """Get LLM instance for the specified provider"""
    if provider not in LLM_CONFIG:
        logger.warning(f"Unknown provider '{provider}', using default {DEFAULT_LLM}")
        provider = DEFAULT_LLM
    
    config = LLM_CONFIG[provider]
    
    if provider == "openai":
        return ChatOpenAI(model=config["model"], temperature=0)
    
    elif provider == "anthropic":
        if ChatAnthropic is None:
            logger.warning("Anthropic provider not installed, falling back to OpenAI")
            return ChatOpenAI(model=LLM_CONFIG[DEFAULT_LLM]["model"], temperature=0)
        return ChatAnthropic(model=config["model"], temperature=0)
    
    elif provider == "google":
        if ChatGoogleGenerativeAI is None:
            logger.warning("Google provider not installed, falling back to OpenAI")
            return ChatOpenAI(model=LLM_CONFIG[DEFAULT_LLM]["model"], temperature=0)
        return ChatGoogleGenerativeAI(model=config["model"], temperature=0)
    
    return ChatOpenAI(model=LLM_CONFIG[DEFAULT_LLM]["model"], temperature=0)


FRAMEWORK_CONFIG = {
    "cypress": {
        "name": "Cypress",
        "file_ext": ".cy.js",
        "default_output": "cypress/e2e",
        "run_cmd": "npx cypress run --spec",
        "code_fence": "javascript",
        "prompt_file_standard": "test_generation_traditional.txt",
        "prompt_file_prompt": "test_generation_prompt_powered.txt",
        "supports_prompt_mode": True,
    },
    "playwright": {
        "name": "Playwright",
        "file_ext": ".spec.ts",
        "default_output": "tests",
        "run_cmd": "npx playwright test",
        "code_fence": "typescript",
        "prompt_file_standard": "test_generation_playwright.txt",
        "supports_prompt_mode": False,
    },
}

# VECTOR STORE - Stores and searches test patterns

class TestPatternStore:
    """Stores test patterns and finds similar ones"""
    
    def __init__(self) -> None:
        logger.info("Setting up vector store")
        
        # Create directory for database
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create vector database
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=self.embeddings
        )
        
        logger.info("Vector store ready")
    
    def store_pattern(self, test_code: str, requirement: str, url: str, test_type: str, filepath: str) -> None:
        """Store a test pattern in the database"""
        logger.info(f"Storing pattern: {requirement}")
        
        # Create a document with the test code
        doc = Document(
            page_content=test_code,
            metadata={
                "requirement": requirement,
                "url": url,
                "test_type": test_type,
                "filepath": filepath,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add to database (auto-persists in new version)
        self.vectorstore.add_documents([doc])
        
        logger.info("Pattern stored")
    
    def search_similar_patterns(self, requirement: str) -> List[Document]:
        """Find patterns similar to the requirement"""
        logger.info(f"Searching for patterns like: {requirement}")
        
        # Check how many patterns we have
        count = self.vectorstore._collection.count()
        
        # Search for similar patterns
        results = []
        
        if count > 0:
            results = self.vectorstore.similarity_search(requirement, k=min(2, count))
        
        logger.info(f"Found {len(results)} similar patterns")
        return results
    
    def get_all_patterns(self) -> List[Document]:
        """Get all stored patterns"""
        count = self.vectorstore._collection.count()
        
        all_patterns = []
        
        # Get patterns if any exist
        if count > 0:
            all_patterns = self.vectorstore.similarity_search("", k=count)
        
        return all_patterns

# STATE - Holds all workflow data

@dataclass
class TestState:
    """State that flows through the workflow"""
    
    # Input data
    requirements: List[str]
    output_dir: str
    use_prompt: bool
    framework: str = "cypress"
    url: Optional[str] = None
    run_tests: bool = False
    llm_provider: str = DEFAULT_LLM
    
    # Processing data
    test_data: Optional[Dict] = None
    context: str = ""
    similar_patterns: List = field(default_factory=list)
    
    # Output data
    generated_tests: List = field(default_factory=list)
    test_results: Optional[Dict] = None
    
    # Vector store
    vector_store: Optional[TestPatternStore] = None

# UTILITIES - Helper functions

def load_prompt_file(filename: str, **variables: Any) -> str:
    """Load a prompt file and fill in variables"""
    logger.info(f"Loading prompt: {filename}")
    
    # Read the prompt file with UTF-8 encoding
    prompt_path = PROMPT_DIR / filename
    prompt_text = prompt_path.read_text(encoding='utf-8')
    
    # Fill in variables
    filled_prompt = prompt_text.format(**variables)
    
    return filled_prompt

# WORKFLOW NODES - Steps in the workflow

def step_1_initialize_vector_store(state: TestState) -> TestState:
    """Step 1: Set up the vector store"""
    logger.info("STEP 1: Initialize Vector Store")
    
    # Create vector store
    state.vector_store = TestPatternStore()
    
    return state


def step_2_fetch_test_data(state: TestState) -> TestState:
    """Step 2: Get test data from URL"""
    logger.info("STEP 2: Fetch Test Data")
    
    url = state.url
    
    # Skip if no URL provided
    if not url:
        logger.info("No URL provided, skipping HTML analysis")
        return state
    
    logger.info(f"Fetching URL: {url}")
    logger.info(f"Using LLM provider: {LLM_CONFIG[state.llm_provider]['name']}")
    
    # Fetch HTML from URL
    response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    html = response.text[:5000]
    
    logger.info(f"Got {len(html)} characters of HTML")
    
    # Analyze HTML with AI
    logger.info("Asking AI to analyze HTML")
    
    llm = get_llm(state.llm_provider)
    prompt = load_prompt_file("html_analysis.txt", url=url, html=html)
    
    ai_response = llm.invoke(prompt)
    content = ai_response.content.strip()
    
    # Extract JSON from response
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    
    test_data = json.loads(content)
    
    # Save test data to file
    filepath = "cypress/fixtures/url_test_data.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Saved test data to: {filepath}")
    
    # Update state
    state.test_data = test_data
    state.context = f"FIXTURE: {filepath}\nURL: {url}\nSELECTORS: {test_data['selectors']}"
    
    return state


def step_3_search_similar_patterns(state: TestState) -> TestState:
    """Step 3: Find similar patterns from past"""
    logger.info("STEP 3: Search Similar Patterns")
    
    # Search for each requirement
    all_patterns = []
    
    for requirement in state.requirements:
        patterns = state.vector_store.search_similar_patterns(requirement)
        all_patterns.extend(patterns)
    
    state.similar_patterns = all_patterns
    
    logger.info(f"Found {len(all_patterns)} similar patterns total")
    
    # Add patterns to context
    has_patterns = len(all_patterns) > 0
    
    pattern_context = "\n\nSIMILAR PATTERNS FROM PAST:\n" + "\n".join([
        f"\nPattern {i}:\n{p.page_content[:200]}..." 
        for i, p in enumerate(all_patterns[:3], 1)
    ]) if has_patterns else ""
    
    state.context = state.context + pattern_context
    
    return state


def step_4_generate_tests(state: TestState) -> TestState:
    """Step 4: Generate test files"""
    logger.info("STEP 4: Generate Tests")
    logger.info(f"Using LLM provider: {LLM_CONFIG[state.llm_provider]['name']}")
    
    # ── Get framework config ──
    fw = FRAMEWORK_CONFIG[state.framework]
    logger.info(f"Framework: {fw['name']}")
    
    # ── Check if --use-prompt is valid for this framework ──
    use_prompt_mode = state.use_prompt and fw["supports_prompt_mode"]
    if state.use_prompt and not fw["supports_prompt_mode"]:
        logger.warning(
            f"--use-prompt ignored: {fw['name']} does not have a prompt feature like cy.prompt(). "
            f"Generating standard {fw['name']} tests instead."
        )
    
    llm = get_llm(state.llm_provider)
    generated_tests = []
    
    # Generate each test
    for index, requirement in enumerate(state.requirements, 1):
        logger.info(f"Generating test {index}/{len(state.requirements)}: {requirement}")
        
        # ── Choose prompt file ──
        if use_prompt_mode:
            prompt_file = fw["prompt_file_prompt"]
        else:
            prompt_file = fw["prompt_file_standard"]
        
        # Load prompt
        prompt = load_prompt_file(prompt_file, requirement=requirement, context=state.context)
        
        # Ask AI to generate test
        ai_response = llm.invoke(prompt)
        content = ai_response.content
        
        # ── Extract code from markdown fences ──
        if "```typescript" in content:
            content = content.split("```typescript")[1].split("```")[0].strip()
        elif "```javascript" in content:
            content = content.split("```javascript")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # ── Prepare output folder ──
        # Cypress: prompt-powered / generated subfolder
        # Playwright: always generated subfolder
        if state.framework == "cypress":
            folder_name = "prompt-powered" if use_prompt_mode else "generated"
        else:
            folder_name = "generated"
        
        output_base = state.output_dir if state.output_dir != "cypress/e2e" else fw["default_output"]
        folder = f"{output_base}/{folder_name}"
        os.makedirs(folder, exist_ok=True)
        
        # ── Create filename with framework-specific extension ──
        slug = re.sub(r'[^\w\s-]', '', requirement.lower()).replace(' ', '-')[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{index:02d}_{slug}_{timestamp}{fw['file_ext']}"
        filepath = f"{folder}/{filename}"
        
        # Save test file
        with open(filepath, 'w') as f:
            f.write(f"// Requirement: {requirement}\n\n{content}")
        
        logger.info(f"Saved: {filename}")
        
        # Store pattern in vector database
        state.vector_store.store_pattern(
            test_code=content,
            requirement=requirement,
            url=state.url or "",
            test_type=f"{state.framework}_{'prompt_powered' if use_prompt_mode else 'traditional'}",
            filepath=filepath
        )
        
        # Add to generated tests list
        test_info = {
            "requirement": requirement,
            "filepath": filepath,
            "filename": filename
        }
        generated_tests.append(test_info)
    
    state.generated_tests = generated_tests
    
    return state


def step_5_run_tests(state: TestState) -> TestState:
    """Step 5: Run the generated tests"""
    logger.info("STEP 5: Run Tests")
    
    fw = FRAMEWORK_CONFIG[state.framework]
    use_prompt_mode = state.use_prompt and fw["supports_prompt_mode"]
    
    # ── Framework-aware test runner ──
    if state.framework == "playwright":
        output_base = state.output_dir if state.output_dir != "cypress/e2e" else fw["default_output"]
        test_path = f"{output_base}/generated"
        cmd = f"npx playwright test {test_path}"
    else:
        folder_name = "prompt-powered" if use_prompt_mode else "generated"
        test_path = f"cypress/e2e/{folder_name}/**/*.cy.js"
        cmd = f"npx cypress run --spec '{test_path}'"
    
    logger.info(f"Running: {cmd}")
    exit_code = os.system(cmd)
    
    # Save results
    state.test_results = {
        "exit_code": exit_code,
        "success": exit_code == 0,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Tests finished with exit code: {exit_code}")
    
    return state

# WORKFLOW - Connects all steps

def should_run_tests(state: TestState) -> str:
    """Check if we should run tests"""
    return "run_tests" if state.run_tests else END


def create_workflow() -> Any:
    """Create the LangGraph workflow"""
    logger.info("Building workflow")
    
    # Create workflow
    workflow = StateGraph(TestState)
    
    # Add steps
    workflow.add_node("step_1", step_1_initialize_vector_store)
    workflow.add_node("step_2", step_2_fetch_test_data)
    workflow.add_node("step_3", step_3_search_similar_patterns)
    workflow.add_node("step_4", step_4_generate_tests)
    workflow.add_node("step_5", step_5_run_tests)
    
    # Connect steps
    workflow.set_entry_point("step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", "step_4")
    
    # Conditional: run tests or end
    workflow.add_conditional_edges(
        "step_4",
        should_run_tests,
        {
            "run_tests": "step_5",
            END: END
        }
    )
    
    workflow.add_edge("step_5", END)
    
    logger.info("Workflow ready")
    
    return workflow.compile()

# ACTIONS - Things the tool can do

def analyze_test_failure(log_text: str) -> str:
    """Analyze why a test failed"""
    logger.info("Analyzing test failure")
    
    # Load the analysis prompt
    prompt_path = PROMPT_DIR / "failure_analysis.txt"
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Format the prompt with the log
    prompt = prompt_template.replace("{log}", log_text)
    
    # Ask AI to analyze
    response = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }
    )
    
    logger.info("Analysis complete")
    return response.json()["choices"][0]["message"]["content"] if response.ok else f"Error: {response.text}"


def list_all_patterns() -> None:
    """Show all stored patterns"""
    logger.info("Listing all patterns")
    
    # Get patterns from database
    store = TestPatternStore()
    patterns = store.get_all_patterns()
    
    logger.info(f"Total patterns: {len(patterns)}")
    
    # Show each pattern
    for i, pattern in enumerate(patterns, 1):
        requirement = pattern.metadata.get('requirement', 'N/A')
        test_type = pattern.metadata.get('test_type', 'N/A')
        preview = pattern.page_content[:100]
        
        logger.info(f"Pattern {i}: {requirement}")
        logger.info(f"  Type: {test_type}")
        logger.info(f"  Preview: {preview}...")


def generate_tests_action(args: argparse.Namespace) -> None:
    """Generate tests using workflow"""
    logger.info("Starting test generation")
    logger.info(f"Framework: {args.framework.upper()}")
    logger.info(f"LLM Provider: {LLM_CONFIG[args.llm]['name']}")
    
    # Create initial state
    state = TestState(
        requirements=args.requirements,
        output_dir=args.out,
        use_prompt=args.use_prompt,
        framework=args.framework,
        url=args.url,
        run_tests=args.run,
        llm_provider=args.llm
    )
    
    # Run workflow
    workflow = create_workflow()
    final_state = workflow.invoke(state)
    
    # Show results
    logger.info("="*50)
    logger.info("GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Framework: {args.framework.upper()}")
    logger.info(f"Generated tests: {len(final_state['generated_tests'])}")
    logger.info(f"Output location: {args.out}")
    logger.info(f"Similar patterns used: {len(final_state['similar_patterns'])}")
    
    logger.info("\nGenerated files:")
    for test in final_state['generated_tests']:
        logger.info(f"  - {test['filename']}")
    
    if final_state.get('test_results'):
        logger.info(f"\nTests passed: {final_state['test_results']['success']}")

# MAIN - Entry point

def main() -> None:
    logger.info("AI-Powered Test Generator (Cypress & Playwright)")
    logger.info("With LangGraph Workflows and Vector Store Learning")
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="AI Test Generator — Cypress & Playwright",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Cypress test
  python qa_automation.py "Test login" --url https://example.com/login

  # Generate Playwright test
  python qa_automation.py "Test login" --url https://example.com/login --framework playwright

  # Use cy.prompt() mode (Cypress only)
  python qa_automation.py "Test login" --url https://example.com/login --use-prompt

  # Generate and run tests
  python qa_automation.py "Test login" --url https://example.com/login --run

  # Use different LLM provider
  python qa_automation.py "Test login" --url https://example.com/login --llm anthropic

  # Analyze test failure
  python qa_automation.py --analyze "CypressError: Element not found"

  # Analyze test failure from file
  python qa_automation.py --analyze -f error.log

  # List stored patterns
  python qa_automation.py --list-patterns
"""
    )
    
    parser.add_argument('requirements', nargs='*', 
                        help='One or more test requirements in natural language (e.g., "Test login with valid credentials")')
    parser.add_argument('--framework', '-fw', choices=['cypress', 'playwright'], default='cypress',
                        help='Target framework: cypress (default) or playwright')
    parser.add_argument('--url', '-u', 
                        help='URL to analyze - fetches page HTML, extracts selectors, and generates test data fixture')
    parser.add_argument('--out', default='cypress/e2e', 
                        help='Output directory for generated tests (default: framework-specific)')
    parser.add_argument('--use-prompt', action='store_true', 
                        help='Generate self-healing tests using cy.prompt() - Cypress only, requires Cypress 15.8.1+ with experimentalCypressPrompt enabled')
    parser.add_argument('--run', action='store_true', 
                        help='Execute tests immediately after generation using framework test runner')
    parser.add_argument('--llm', choices=list(LLM_CONFIG.keys()), default=DEFAULT_LLM,
                        help=f'LLM provider to use for test generation: {" or ".join(LLM_CONFIG.keys())} (default: {DEFAULT_LLM})')
    parser.add_argument('--analyze', '-a', nargs='?', const='', 
                        help='Analyze test failure log using AI - provide error message directly or use with --file')
    parser.add_argument('--file', '-f', 
                        help='Path to log file containing test failure output to analyze')
    parser.add_argument('--list-patterns', action='store_true', 
                        help='Display all test patterns stored in the vector database')
    
    args = parser.parse_args()
    
    # Determine what to do
    analyze_mode = args.analyze is not None or args.file
    list_mode = args.list_patterns
    generate_mode = len(args.requirements) > 0
    
    # Do the action
    if analyze_mode:
        log_text = open(args.file).read() if args.file else args.analyze or sys.stdin.read()
        result = analyze_test_failure(log_text)
        logger.info(result)
        return
    
    if list_mode:
        list_all_patterns()
        return
    
    if generate_mode:
        generate_tests_action(args)
        return
    
    # Show help
    parser.print_help()


if __name__ == "__main__":

    main()