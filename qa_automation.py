#!/usr/bin/env python3
"""
AI-Powered Cypress Test Generator with LangGraph & Vector Store
"""

import os
import re
import sys
import json
import argparse
import requests
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*PyTorch.*')
warnings.filterwarnings('ignore', message='.*TensorFlow.*')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"
VECTOR_DB_DIR = Path(__file__).parent / "vector_db"

# VECTOR STORE - Stores and searches test patterns

class TestPatternStore:
    """Stores test patterns and finds similar ones"""
    
    def __init__(self):
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
    
    def store_pattern(self, test_code, requirement, url, test_type, filepath):
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
    
    def search_similar_patterns(self, requirement):
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
    
    def get_all_patterns(self):
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
    url: Optional[str] = None
    run_tests: bool = False
    
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

def load_prompt_file(filename, **variables):
    """Load a prompt file and fill in variables"""
    logger.info(f"Loading prompt: {filename}")
    
    # Read the prompt file with UTF-8 encoding
    prompt_path = PROMPT_DIR / filename
    prompt_text = prompt_path.read_text(encoding='utf-8')
    
    # Fill in variables
    filled_prompt = prompt_text.format(**variables)
    
    return filled_prompt

# WORKFLOW NODES - Steps in the workflow

def step_1_initialize_vector_store(state):
    """Step 1: Set up the vector store"""
    logger.info("STEP 1: Initialize Vector Store")
    
    # Create vector store
    state.vector_store = TestPatternStore()
    
    return state


def step_2_fetch_test_data(state):
    """Step 2: Get test data from URL"""
    logger.info("STEP 2: Fetch Test Data")
    
    url = state.url
    logger.info(f"Fetching URL: {url}")
    
    # Fetch HTML from URL
    response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    html = response.text[:5000]
    
    logger.info(f"Got {len(html)} characters of HTML")
    
    # Analyze HTML with AI
    logger.info("Asking AI to analyze HTML")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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


def step_3_search_similar_patterns(state):
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


def step_4_generate_tests(state):
    """Step 4: Generate test files"""
    logger.info("STEP 4: Generate Tests")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    generated_tests = []
    
    # Generate each test
    for index, requirement in enumerate(state.requirements, 1):
        logger.info(f"Generating test {index}/{len(state.requirements)}: {requirement}")
        
        # Choose prompt file
        prompt_file = "test_generation_prompt_powered.txt" if state.use_prompt else "test_generation_traditional.txt"
        
        # Load prompt
        prompt = load_prompt_file(prompt_file, requirement=requirement, context=state.context)
        
        # Ask AI to generate test
        ai_response = llm.invoke(prompt)
        content = ai_response.content
        
        # Extract JavaScript code
        if "```javascript" in content:
            content = content.split("```javascript")[1].split("```")[0].strip()
        
        # Prepare file paths
        folder_name = "prompt-powered" if state.use_prompt else "generated"
        folder = f"{state.output_dir}/{folder_name}"
        os.makedirs(folder, exist_ok=True)
        
        # Create filename
        slug = re.sub(r'[^\w\s-]', '', requirement.lower()).replace(' ', '-')[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{index:02d}_{slug}_{timestamp}.cy.js"
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
            test_type="prompt_powered" if state.use_prompt else "traditional",
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


def step_5_run_tests(state):
    """Step 5: Run the generated tests"""
    logger.info("STEP 5: Run Tests")
    
    # Build test path
    folder_name = "prompt-powered" if state.use_prompt else "generated"
    test_path = f"cypress/e2e/{folder_name}/**/*.cy.js"
    
    logger.info(f"Running tests: {test_path}")
    
    # Run Cypress
    exit_code = os.system(f"npx cypress run --spec '{test_path}'")
    
    # Save results
    state.test_results = {
        "exit_code": exit_code,
        "success": exit_code == 0,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Tests finished with exit code: {exit_code}")
    
    return state

# WORKFLOW - Connects all steps

def should_run_tests(state):
    """Check if we should run tests"""
    return "run_tests" if state.run_tests else END


def create_workflow():
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

def analyze_test_failure(log_text):
    """Analyze why a test failed"""
    logger.info("Analyzing test failure")
    
    # Ask AI to analyze
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": f"Analyze this Cypress test failure. Reply ONLY:\nREASON: (one line)\nFIX: (one line)\n\n{log_text}"}],
            "max_tokens": 150
        }
    )
    
    logger.info("Analysis complete")
    return response.json()["choices"][0]["message"]["content"] if response.ok else f"Error: {response.text}"


def list_all_patterns():
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


def generate_tests_action(args):
    """Generate tests using workflow"""
    logger.info("Starting test generation")
    
    # Create initial state
    state = TestState(
        requirements=args.requirements,
        output_dir=args.out,
        use_prompt=args.use_prompt,
        url=args.url,
        run_tests=args.run
    )
    
    # Run workflow
    workflow = create_workflow()
    final_state = workflow.invoke(state)
    
    # Show results
    logger.info("="*50)
    logger.info("GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Generated tests: {len(final_state['generated_tests'])}")
    logger.info(f"Output location: {args.out}")
    logger.info(f"Similar patterns used: {len(final_state['similar_patterns'])}")
    
    logger.info("\nGenerated files:")
    for test in final_state['generated_tests']:
        logger.info(f"  - {test['filename']}")
    
    if final_state.get('test_results'):
        logger.info(f"\nTests passed: {final_state['test_results']['success']}")

# MAIN - Entry point

def main():
    logger.info("AI-Powered Cypress Test Generator")
    logger.info("With LangGraph Workflows and Vector Store Learning")
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="AI Cypress Test Generator")
    
    parser.add_argument('--analyze', '-a', nargs='?', const='', help='Analyze test failure')
    parser.add_argument('--file', '-f', help='Log file to analyze')
    parser.add_argument('requirements', nargs='*', help='Test requirements')
    parser.add_argument('--out', default='cypress/e2e', help='Output directory')
    parser.add_argument('--use-prompt', action='store_true', help='Use cy.prompt()')
    parser.add_argument('--run', action='store_true', help='Run tests after generation')
    parser.add_argument('--url', '-u', help='URL to analyze')
    parser.add_argument('--list-patterns', action='store_true', help='List stored patterns')
    
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