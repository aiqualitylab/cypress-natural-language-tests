# Cypress test generation framework driven by natural-language requirements using AI

An AI-powered tool that generates Cypress end-to-end tests from natural language requirements using OpenAI's GPT-4 and LangGraph workflows.

## Features

* 🤖 **AI-Powered**: Converts natural language requirements into working Cypress tests
* 📚 **Document Context**: Optional vector store integration for additional context from documentation
* 🔄 **Workflow Management**: Uses LangGraph for structured test generation pipeline
* ⚡ **Auto-Run**: Optionally runs generated tests immediately with Cypress
* 📁 **Organized Output**: Generates timestamped, descriptively named test files

## Prerequisites

* **Node.js** (v14 or higher)
* **Python** (v3.8 or higher)
* **OpenAI API Key**
* **Cypress** (installed in your project)

## Installation

### 1. Clone and Setup Python Environment

```bash
git clone https://github.com/aiqualitylab/cypress-natural-language-tests.git
cd cypress-natural-language-tests
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
langgraph
langchain-openai
langchain-community
langchain
chromadb
python-dotenv
```

### 3. Setup Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Initialize Cypress (if not already done)

```bash
npm install cypress --save-dev
npx cypress open  # Initial setup
```

## Usage

### Basic Usage

Generate Cypress tests from natural language requirements:

```bash
python qa_automation.py "Test user login with valid credentials" "Test login with invalid credentials"
```

### Command Line Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `requirements` | One or more test requirements (positional) | Required |
| `--out` | Output directory for generated specs | `cypress/e2e` |
| `--run` | Run Cypress tests after generation | `false` |
| `--docs` | Directory with additional context documents | `None` |
| `--persist-vstore` | Create/update vector store from docs | `false` |

## Examples

### 1. Basic Test Generation

```bash
python qa_automation.py "Test login page"
```

**Output**: `cypress/e2e/01_test-login-page-loads-correctly_20240304_143022.cy.js`

### 2. Multiple Requirements

```bash
python qa_automation.py \
  "Test successful login with valid credentials" \
  "Test login failure with invalid password" \
  "Test login form validation for empty fields"
```

### 3. With Documentation Context

```bash
python qa_automation.py \
  "Test the checkout process" \
  --docs ./api-docs \
  --persist-vstore
```

### 4. Generate and Run Tests

```bash
python qa_automation.py \
  "Test user profile update" \
  --run \
  --out cypress/e2e/profile
```

## Generated Test Structure

Each generated test follows this structure:

```javascript
// Requirement: Test user login with valid credentials
describe('User Login', () => {
  it('should login successfully with valid credentials', () => {
    cy.visit('https://the-internet.herokuapp.com/login');
    cy.get('#username').type('tomsmith');
    cy.get('#password').type('SuperSecretPassword!');
    cy.get('button[type="submit"]').click();
    cy.get('.flash.success').should('contain', 'You logged into a secure area!');
  });
  
  it('should show error with invalid credentials', () => {
    cy.visit('https://the-internet.herokuapp.com/login');
    cy.get('#username').type('invaliduser');
    cy.get('#password').type('wrongpassword');
    cy.get('button[type="submit"]').click();
    cy.get('.flash.error').should('contain', 'Your username is invalid!');
  });
});
```

## File Naming Convention

Generated files follow this pattern:

```
{sequence}_{slugified-requirement}_{timestamp}.cy.js
```

Example: `01_test-user-login_20240304_143022.cy.js`

## Vector Store Integration

When using `--docs` and `--persist-vstore`, the tool:

1. Indexes all documents in the specified directory
2. Creates/updates a Chroma vector store in `./vector_store`
3. Uses document context to improve test generation accuracy

**Supported document formats**: `.txt`, `.md`, `.json`, `.js`, `.html`, etc.

## Configuration

### Customizing the Base URL

Edit the `CY_PROMPT_TEMPLATE` in `qa_automation.py` to change the default base URL:

```python
- Use `cy.visit('https://your-app.com')` as the base URL.
```

### Adjusting LLM Settings

Modify the LLM configuration:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
    temperature=0  # 0 for deterministic, higher for creativity
)
```

## Workflow Architecture

The tool uses LangGraph to orchestrate the following steps:

1. **ParseCLI** - Parse command line arguments
2. **BuildVectorStore** - Index documentation (if provided)
3. **GenerateTests** - Create Cypress tests using AI
4. **RunCypress** - Execute tests (if requested)

## GitHub Copilot Integration

This repository includes GitHub Copilot instructions and skills to help you work more efficiently with AI assistance. See [GITHUB_COPILOT_SETUP.md](.github/GITHUB_COPILOT_SETUP.md) for details.

### Available Copilot Skills

- **Cypress AI Test Generation**: Specialized knowledge for working with the test generation pipeline
- **LangGraph Workflow Development**: Comprehensive guide for building and debugging workflows

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Related Projects

Check out more AI-powered testing projects at [@aiqualitylab](https://github.com/aiqualitylab)

## About

This project generates Cypress E2E tests automatically from natural language requirements using OpenAI GPT, LangChain, and LangGraph.

**Author**: ([@aiqualitylab](https://github.com/aiqualitylab))  
**Medium**: [AQE Publication](https://medium.com/ai-in-quality-assurance)
