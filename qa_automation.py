#!/usr/bin/env python3
"""
Hybrid Cypress Test Generator with cy.prompt() Integration
Combines LangGraph orchestration with Cypress's native AI capabilities
AI Failure Analyzer (OpenRouter LLM)
"""

import os
import re
import sys
import argparse
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# FAILURE ANALYZER 

def analyze_failure(log: str) -> str:
    """Send log to OpenRouter free LLM, get reason + fix."""
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": [{"role": "user", "content": f"Analyze this Cypress test failure. Reply ONLY:\nREASON: (one line)\nFIX: (one line)\n\n{log}"}],
            "max_tokens": 150
        }
    )
    return response.json()["choices"][0]["message"]["content"] if response.ok else f"Error: {response.text}"


# URL-BASED TEST DATA GENERATOR (GENERIC FOR ANY URL!)

def generate_test_data_from_url(url: str, requirements: list) -> tuple:
    """
    Fetch any URL, analyze HTML, generate test data.
    Returns: (context_string, json_data, filepath)
    """
    import json as json_module
    
    # Fetch page
    print(f"   Fetching {url}...")
    try:
        resp = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if not resp.ok:
            return (f"HTTP {resp.status_code} - Could not fetch URL", None, None)
        html = resp.text[:5000]
        print(f"   Fetched {len(resp.text)} bytes")
    except requests.exceptions.Timeout:
        return ("Connection timed out - URL may be unreachable", None, None)
    except requests.exceptions.ConnectionError:
        return ("Connection failed - check URL and network", None, None)
    except Exception as e:
        return (f"Fetch error: {e}", None, None)
    
    # Check if HTML has form elements
    if '<form' not in html.lower() and '<input' not in html.lower():
        print(f"   Warning: No form elements detected in HTML")
    
    # Analyze with AI
    print(f"   Analyzing with AI...")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception as e:
        return (f"OpenAI API error: {e} - Check OPENAI_API_KEY", None, None)
    
    prompt = f"""Analyze this HTML and generate test data for automation.

URL: {url}
HTML:
{html}

Return ONLY valid JSON:
{{"url": "{url}",
  "selectors": {{"field1": "#selector1", "field2": "#selector2", "submit": "button selector"}},
  "test_cases": [
    {{"name": "valid_test", "field1": "valid_value", "field2": "valid_value", "expected": "success"}},
    {{"name": "invalid_test", "field1": "wrong", "field2": "wrong", "expected": "error"}}
  ]
}}

Rules:
- Use REAL selectors from HTML (#id, .class, [name=x])
- Use field names that match form inputs (username, password, email, etc.)
- test_cases MUST have names: "valid_test" and "invalid_test"
- NO empty values
- Return ONLY JSON"""

    try:
        content = llm.invoke(prompt).content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").split("```")[0].strip()
        
        test_data = json_module.loads(content)
        
        # Save fixture
        os.makedirs("cypress/fixtures", exist_ok=True)
        filepath = "cypress/fixtures/url_test_data.json"
        with open(filepath, 'w') as f:
            json_module.dump(test_data, f, indent=2)
        print(f"   Saved: {filepath}")
        
        context = f"""
FIXTURE: cypress/fixtures/url_test_data.json
URL: {url}
SELECTORS: {test_data.get('selectors', {})}
TEST_CASES: {test_data.get('test_cases', [])}

Use cy.fixture('url_test_data.json') with function() and this.testData"""
        
        return (context, test_data, filepath)
        
    except json_module.JSONDecodeError as e:
        return (f"AI returned invalid JSON: {e}", None, None)
    except Exception as e:
        return (f"AI analysis error: {e}", None, None)


# TEST GENERATION

@dataclass
class TestGenerationState:
    """State for test generation workflow"""
    requirements: List[str]
    output_dir: str
    use_prompt: bool
    docs_context: Optional[str]
    generated_tests: List[Dict[str, Any]]
    run_tests: bool
    error: Optional[str]


class HybridTestGenerator:
    """Generates Cypress tests with cy.prompt() integration"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings()
        
    def create_traditional_test_prompt(self) -> str:
        """Prompt for generating traditional Cypress tests"""
        return """You are an expert Cypress test automation engineer.

Generate a complete Cypress test file based on the requirement.

REQUIREMENT: {requirement}

CONTEXT (if available): {context}

GUIDELINES:
- Use Cypress best practices
- Use DYNAMIC selectors from this.testData.selectors (NOT hardcoded!)
- Generate TWO test cases: valid + invalid (NOT empty)
- Use cy.fixture() with this context pattern
- Return ONLY runnable JavaScript code

USE THIS PATTERN (FULLY DYNAMIC - works for ANY URL):

```javascript
describe('Tests', function () {{
    beforeEach(function () {{
        cy.fixture('url_test_data').then((data) => {{
            this.testData = data;
        }});
    }});

    it('should succeed with valid data', function () {{
        cy.visit(this.testData.url);
        const valid = this.testData.test_cases.find(tc => tc.name === 'valid_test');
        const selectors = this.testData.selectors;
        
        // DYNAMIC: Loop through all selectors from fixture
        Object.keys(selectors).forEach(field => {{
            if (field !== 'submit' && valid[field]) {{
                cy.get(selectors[field]).type(valid[field]);
            }}
        }});
        
        cy.get(selectors.submit).click();
    }});

    it('should fail with invalid data', function () {{
        cy.visit(this.testData.url);
        const invalid = this.testData.test_cases.find(tc => tc.name === 'invalid_test');
        const selectors = this.testData.selectors;
        
        Object.keys(selectors).forEach(field => {{
            if (field !== 'submit' && invalid[field]) {{
                cy.get(selectors[field]).type(invalid[field]);
            }}
        }});
        
        cy.get(selectors.submit).click();
    }});
}});
```

CRITICAL:
- Use function() NOT arrow =>
- cy.visit(this.testData.url) - URL from fixture!
- cy.get(selectors[field]) - selectors from fixture!
- DO NOT hardcode #username, #password, or any selector!
- DO NOT hardcode any URL!

Generate ONLY the test code, no explanations."""

    def create_prompt_powered_test_prompt(self) -> str:
        """Prompt for generating cy.prompt() enabled tests"""
        return """You are an expert Cypress test automation engineer with cy.prompt() expertise.

Generate a Cypress test file using cy.prompt() for self-healing capabilities.

REQUIREMENT: {requirement}

CONTEXT (if available): {context}

GUIDELINES FOR cy.prompt():
- Use cy.prompt() with natural language step arrays
- Each step should be clear and descriptive
- Include verification steps
- Use natural language like "Visit the page", "Click the submit button"
- Group related steps logically
- Use fixture data for URL and test data (NOT hardcoded!)

DYNAMIC PATTERN (works for ANY URL):

```javascript
describe('Tests', function () {{
    beforeEach(function () {{
        cy.fixture('url_test_data').then((data) => {{
            this.testData = data;
        }});
    }});

    it('should succeed with valid data', function () {{
        cy.visit(this.testData.url);
        const valid = this.testData.test_cases.find(tc => tc.name === 'valid_test');
        const selectors = this.testData.selectors;
        
        // Fill form fields dynamically
        Object.keys(selectors).forEach(field => {{
            if (field !== 'submit' && valid[field]) {{
                cy.get(selectors[field]).type(valid[field]);
            }}
        }});
        
        cy.get(selectors.submit).click();
    }});
}});
```

CRITICAL:
- Use this.testData.url for visiting (NOT hardcoded URL!)
- Use this.testData.selectors for cy.get() (NOT hardcoded selectors!)
- Use this.testData.test_cases for test values

Generate ONLY the test code, no explanations."""

    def generate_test_content(
        self, 
        requirement: str, 
        context: str = "", 
        use_prompt: bool = False
    ) -> str:
        """Generate test content using AI"""
        
        # Choose the right prompt template
        if use_prompt:
            template = self.create_prompt_powered_test_prompt()
        else:
            template = self.create_traditional_test_prompt()
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        response = chain.invoke({
            "requirement": requirement,
            "context": context or "No additional context provided"
        })
        
        # Extract code from response
        content = response.content
        
        # Remove markdown code blocks if present
        if "```javascript" in content:
            content = content.split("```javascript")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return content

    def slugify(self, text: str) -> str:
        """Convert text to slug format"""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text[:50]  # Limit length

    def save_test_file(
        self, 
        content: str, 
        requirement: str, 
        output_dir: str,
        use_prompt: bool,
        index: int
    ) -> Dict[str, Any]:
        """Save generated test to file"""
        
        # Simple: Choose folder based on test type
        if use_prompt:
            folder = f"{output_dir}/prompt-powered"
        else:
            folder = f"{output_dir}/generated"
        
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Simple filename
        slug = self.slugify(requirement)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{index:02d}_{slug}_{timestamp}.cy.js"
        filepath = f"{folder}/{filename}"
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(f"// Requirement: {requirement}\n")
            f.write(f"// Test Type: {'cy.prompt()' if use_prompt else 'Traditional'}\n\n")
            f.write(content)
        
        return {
            "requirement": requirement,
            "filepath": filepath,
            "filename": filename
        }


class DocumentContextLoader:
    """Load and process documentation for context"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_documents(self, docs_dir: str) -> List[Document]:
        """Load documents from directory"""
        docs = []
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            return docs
            
        for file_path in docs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.json']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": str(file_path)}
                        ))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
        return docs
    
    def create_vector_store(self, docs: List[Document], persist_dir: str = "./vector_store"):
        """Create vector store from documents"""
        if not docs:
            return None
            
        splits = self.text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        return vector_store
    
    def get_relevant_context(self, vector_store, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        if not vector_store:
            return ""
            
        results = vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context


def parse_cli_node(state: TestGenerationState) -> TestGenerationState:
    """Parse CLI arguments - initial node"""
    test_type = "cy.prompt()" if state.use_prompt else "Traditional"
    print(f"Generating {len(state.requirements)} test(s) - Type: {test_type}")
    return state


def load_context_node(state: TestGenerationState) -> TestGenerationState:
    """Load documentation context if provided"""
    # This is a placeholder - context loading happens in main
    return state


def generate_tests_node(state: TestGenerationState) -> TestGenerationState:
    """Generate test files"""
    generator = HybridTestGenerator()
    generated = []
    
    for idx, requirement in enumerate(state.requirements, 1):
        print(f"\nGenerating test {idx}/{len(state.requirements)}...")
        
        try:
            # Generate test
            content = generator.generate_test_content(
                requirement=requirement,
                context=state.docs_context or "",
                use_prompt=state.use_prompt
            )
            
            # Save test
            result = generator.save_test_file(
                content=content,
                requirement=requirement,
                output_dir=state.output_dir,
                use_prompt=state.use_prompt,
                index=idx
            )
            
            generated.append(result)
            print(f"Saved: {result['filename']}")
            
        except Exception as e:
            print(f"Error: {e}")
            state.error = str(e)
    
    state.generated_tests = generated
    return state


def run_cypress_node(state: TestGenerationState) -> TestGenerationState:
    """Run Cypress tests if requested"""
    if not state.run_tests or not state.generated_tests:
        return state
    
    print("\nRunning tests...")
    
    # Choose which tests to run
    if state.use_prompt:
        tests = "cypress/e2e/prompt-powered/**/*.cy.js"
    else:
        tests = "cypress/e2e/generated/**/*.cy.js"
    
    # Run Cypress
    cmd = f"npx cypress run --spec '{tests}'"
    
    try:
        os.system(cmd)
        print("Tests completed")
    except Exception as e:
        print(f"Error: {e}")
        state.error = str(e)
    
    return state


def create_workflow() -> StateGraph:
    """Create LangGraph workflow"""
    workflow = StateGraph(TestGenerationState)
    
    # Add nodes
    workflow.add_node("parse_cli", parse_cli_node)
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("generate_tests", generate_tests_node)
    workflow.add_node("run_cypress", run_cypress_node)
    
    # Define edges
    workflow.set_entry_point("parse_cli")
    workflow.add_edge("parse_cli", "load_context")
    workflow.add_edge("load_context", "generate_tests")
    workflow.add_edge("generate_tests", "run_cypress")
    workflow.add_edge("run_cypress", END)
    
    return workflow.compile()


def main():
    parser = argparse.ArgumentParser(
        description="Cypress Test Generator & Failure Analyzer",
        epilog="""
EXAMPLES:
  Traditional:      python qa_automation.py "Test login" --url https://example.com/login
  From JSON:        python qa_automation.py "Test login" --data testdata.json
  Analyze failure:  python qa_automation.py --analyze "CypressError: Element not found"
        """
    )
    
    # Analyze mode
    parser.add_argument('--analyze', '-a', nargs='?', const='', help='Analyze failure')
    parser.add_argument('--file', '-f', help='Log file to analyze')
    
    # Generate mode
    parser.add_argument('requirements', nargs='*', help='What to test')
    parser.add_argument('--out', default='cypress/e2e', help='Output folder')
    parser.add_argument('--use-prompt', action='store_true', help='Enable self-healing tests')
    parser.add_argument('--run', action='store_true', help='Run tests after generation')
    parser.add_argument('--docs', help='Documentation folder for context')
    
    # Test data options
    parser.add_argument('--url', '-u', help='Live URL to fetch and generate test data')
    parser.add_argument('--data', '-d', help='JSON file with test data')
    
    args = parser.parse_args()
    
    # === ANALYZE MODE ===
    if args.analyze is not None or args.file:
        if args.file:
            with open(args.file) as f:
                log = f.read()
        elif args.analyze:
            log = args.analyze
        elif not sys.stdin.isatty():
            log = sys.stdin.read()
        else:
            print("Usage: --analyze 'error' or -f log.txt")
            sys.exit(1)
        print("\nAnalyzing...\n")
        print(analyze_failure(log))
        return
    
    # === GENERATE MODE ===
    if not args.requirements:
        parser.print_help()
        return
    
    test_data_context = ""
    saved_test_data_file = None
    
    # Option 1: Fetch REAL live URL
    if args.url:
        print(f"Fetching live URL: {args.url}")
        context, test_data, filepath = generate_test_data_from_url(args.url, args.requirements)
        
        if filepath:
            test_data_context += context
            saved_test_data_file = filepath
            print(f"   Test data generated successfully")
        else:
            print(f"   ERROR: {context}")
            print(f"   Failed to generate test data from URL")
            print(f"   Try: --data option with existing JSON file instead")
            sys.exit(1)
    
    # Option 2: Load from JSON file
    if args.data:
        try:
            import json as json_module
            with open(args.data, 'r') as f:
                test_data = json_module.load(f)
            test_data_context += f"\n\nTEST DATA FROM FILE:\n```json\n{json_module.dumps(test_data, indent=2)}\n```"
            print(f"Loaded test data from: {args.data}")
            saved_test_data_file = args.data
        except Exception as e:
            print(f"Could not load test data: {e}")
    
    # Load documentation context if provided
    docs_context = None
    if args.docs:
        print(f"Loading documentation from: {args.docs}")
        loader = DocumentContextLoader(OpenAIEmbeddings())
        docs = loader.load_documents(args.docs)
        if docs:
            vector_store = loader.create_vector_store(docs)
            combined_query = " ".join(args.requirements)
            docs_context = loader.get_relevant_context(vector_store, combined_query)
            print(f"Loaded context from {len(docs)} document(s)")
    
    # Combine contexts
    full_context = (docs_context or "") + test_data_context
    
    # Create initial state
    initial_state = TestGenerationState(
        requirements=args.requirements,
        output_dir=args.out,
        use_prompt=args.use_prompt,
        docs_context=full_context if full_context else None,
        generated_tests=[],
        run_tests=args.run,
        error=None
    )
    
    # Run workflow
    print("\n" + "="*50)
    print("Starting Test Generation")
    print("="*50)
    
    workflow = create_workflow()
    result = workflow.invoke(initial_state)
    
    # Print summary
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    
    test_count = len(result['generated_tests'])
    if args.use_prompt:
        test_type = "cy.prompt()"
    else:
        test_type = "Traditional"
    
    print(f"Generated: {test_count} test(s)")
    print(f"Type: {test_type}")
    print(f"Location: {args.out}")
    
    if saved_test_data_file:
        print(f"\nTest Data: {saved_test_data_file}")
    
    if result['generated_tests']:
        print("\nFiles:")
        for test in result['generated_tests']:
            print(f"  - {test['filename']}")
    
    if result['error']:
        print(f"\nError: {result['error']}")
    
    print("\nNext step:")
    if args.use_prompt:
        print("  npm run cypress:run:prompt")
    else:
        print("  npm run cypress:run:traditional")


if __name__ == "__main__":
    main()