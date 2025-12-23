#!/usr/bin/env python3
"""
Hybrid Cypress Test Generator with cy.prompt() Integration
Combines LangGraph orchestration with Cypress's native AI capabilities
"""

import os
import re
import json
import argparse
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
- Use Cypress best practices.
- Use `describe` and `it` blocks.
- Prefer **real, working selectors from the page** (id, class, name) over non-existent data-testid.
- Include clear assertions for both success and error messages.
- Handle forms, buttons, and navigation.
- Use `cy.visit('https://the-internet.herokuapp.com/login')` as the base URL.
- Include both a positive and a negative path when applicable.
- Do not include explanations or markdown; return ONLY runnable JavaScript code.
- Ensure the code is ready to run in a standard Cypress setup.

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
- Use natural language like "Visit the login page", "Click the submit button"
- Group related steps logically
- Add fallback traditional Cypress commands for critical assertions

EXAMPLE STRUCTURE:
```javascript
describe('User Login Tests', () => {{
    const baseUrl = 'https://the-internet.herokuapp.com/login';

    beforeEach(() => {{
        cy.visit(baseUrl);
    }});

    it('should successfully log in with valid credentials', () => {{
        cy.get('input[type="text"]').type('tomsmith');
        cy.get('input[type="password"]').type('SuperSecretPassword!');
        cy.get('button[type="submit"]').click();

        cy.url().should('include', '/secure');
        cy.get('.flash.success').should('be.visible').and('contain', 'You logged into a secure area!');
    }});
}});
```

Generate ONLY the test code using cy.prompt(), no explanations."""

    def generate_test_content(
        self, 
        requirement: str, 
        context: str = "", 
        use_prompt: bool = False
    ) -> str:
        """Generate test content using AI"""
        
        template = (
            self.create_prompt_powered_test_prompt() 
            if use_prompt 
            else self.create_traditional_test_prompt()
        )
        
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
        print(f"\n Generating test {idx}/{len(state.requirements)}...")
        
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
    
    print("\n Running tests...")
    
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
    parser = argparse.ArgumentParser(description="Generate Cypress tests")
    
    # Required: Test requirements
    parser.add_argument('requirements', nargs='+', help='What to test')
    
    # Optional: Where to save
    parser.add_argument('--out', default='cypress/e2e', help='Output folder')
    
    # Optional: Use cy.prompt() for self-healing
    parser.add_argument('--use-prompt', action='store_true', help='Enable self-healing tests')
    
    # Optional: Run tests after generating
    parser.add_argument('--run', action='store_true', help='Run tests after generation')
    
    # Optional: Add documentation context
    parser.add_argument('--docs', help='Documentation folder for context')
    
    args = parser.parse_args()
    
    # Load documentation context if provided
    docs_context = None
    if args.docs:
        print(f"Loading documentation from: {args.docs}")
        loader = DocumentContextLoader(OpenAIEmbeddings())
        docs = loader.load_documents(args.docs)
        if docs:
            vector_store = loader.create_vector_store(docs)
            # Get context for all requirements combined
            combined_query = " ".join(args.requirements)
            docs_context = loader.get_relevant_context(vector_store, combined_query)
            print(f"Loaded context from {len(docs)} document(s)")
    
    # Create initial state
    initial_state = TestGenerationState(
        requirements=args.requirements,
        output_dir=args.out,
        use_prompt=args.use_prompt,
        docs_context=docs_context,
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
    test_type = "cy.prompt()" if args.use_prompt else "Traditional"
    
    print(f"Generated: {test_count} test(s)")
    print(f"Type: {test_type}")
    print(f"Location: {args.out}")
    
    if result['generated_tests']:
        print("\nFiles:")
        for test in result['generated_tests']:
            print(f"  - {test['filename']}")
    
    if result['error']:
        print(f"\n Error: {result['error']}")
    
    print("\nNext step:")
    if args.use_prompt:
        print("  npm run cypress:run:prompt")
    else:
        print("  npm run cypress:run:traditional")


if __name__ == "__main__":
    main()