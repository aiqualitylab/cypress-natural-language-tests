# Cypress Natural Language Test Generation Framework

## Project Overview

This repository contains an AI-powered test automation framework that generates Cypress end-to-end tests from natural language requirements. The system uses OpenAI's GPT-4, LangChain, and LangGraph to orchestrate an automated workflow that converts plain English test requirements into executable Cypress test specifications.

## Tech Stack

### Core Technologies
- **Python 3.8+**: Primary orchestration language for the test generation pipeline
- **Node.js 14+**: Runtime for Cypress test execution
- **Cypress**: End-to-end testing framework for generated tests
- **OpenAI GPT-4**: Large language model for natural language to code conversion
- **LangChain**: Framework for LLM application development
- **LangGraph**: Workflow orchestration and state management
- **ChromaDB**: Vector database for document context storage

### Key Dependencies
- `langgraph` - Workflow graph management
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community tools and utilities
- `chromadb` - Vector store for documentation context
- `python-dotenv` - Environment variable management

## Project Structure

```
cypress-natural-language-tests/
├── .env                          # Environment variables (OPENAI_API_KEY)
├── .github/
│   ├── copilot-instructions.md   # This file
│   └── skills/                   # Agent skills directory
├── cypress/
│   └── e2e/                      # Generated Cypress test specifications
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies
├── qa_automation.py              # Main orchestration script
└── README.md                     # Project documentation
```

## Coding Guidelines

### Python Code Standards
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Keep functions focused on single responsibilities
- Use descriptive variable names that reflect the AI/testing domain (e.g., `test_requirements`, `generated_spec`)
- Add docstrings to all classes and public functions explaining their role in the test generation pipeline

### Generated Cypress Test Standards
- All generated tests must use the Describe-It pattern
- Include meaningful test descriptions that mirror the natural language requirement
- Use Cypress best practices:
  - `cy.visit()` for navigation
  - Proper selectors (prefer `data-testid` attributes when available)
  - Clear assertions with `.should()` commands
  - Appropriate waiting strategies
- Follow the naming convention: `{sequence}_{slugified-requirement}_{timestamp}.cy.js`
- Each test file should be self-contained and executable independently

### LangGraph Workflow Standards
- Define clear state schemas for the workflow
- Use type annotations for state transitions
- Keep node functions pure and focused on single transformations
- Include error handling at each workflow node
- Log state transitions for debugging

### Environment & Configuration
- Never commit `.env` files or API keys
- Use `.env.example` for documenting required environment variables
- OpenAI API key is required and must be set in `.env`
- Default base URL for tests can be customized in `CY_PROMPT_TEMPLATE`

## Development Workflow

### Adding New Features
1. Modify the LangGraph workflow in `qa_automation.py`
2. Update the state schema if adding new data flow
3. Adjust the LLM prompt template (`CY_PROMPT_TEMPLATE`) for test generation changes
4. Test with various natural language inputs to ensure robust generation
5. Update documentation in README.md

### Testing Generated Code
1. Generate tests using: `python qa_automation.py "Your requirement"`
2. Review generated files in `cypress/e2e/`
3. Run tests manually: `npx cypress run` or with `--run` flag
4. Validate test quality and adjust prompts if needed

### Vector Store Management
- Vector store is created in `./vector_store` directory
- Use `--docs` flag to provide additional context from documentation
- Use `--persist-vstore` flag to create/update the vector database
- Supported document formats: `.txt`, `.md`, `.json`, `.js`, `.html`

## AI-Specific Considerations

### When Working with LLM Prompts
- The `CY_PROMPT_TEMPLATE` is critical for test quality
- When modifying prompts:
  - Maintain the instruction to use Cypress best practices
  - Keep the base URL directive clear
  - Preserve the requirement reference format
  - Test with diverse requirements after changes

### When Modifying the LangGraph Workflow
- The workflow follows: CLI → VectorStore → TestGeneration → CypressExecution
- State passes through nodes sequentially
- Each node should validate its inputs and handle errors gracefully
- Add new nodes only when they represent distinct processing steps

### When Enhancing AI Capabilities
- Consider if additional context improves test quality
- Balance context window size with generation speed
- Use vector store for large documentation sets
- Monitor token usage and costs when adjusting model parameters

## Common Tasks

### Generate Tests from Requirements
```bash
python qa_automation.py "Test user login" "Test form validation"
```

### Generate with Documentation Context
```bash
python qa_automation.py "Test checkout flow" --docs ./api-docs --persist-vstore
```

### Generate and Execute Tests
```bash
python qa_automation.py "Test navigation" --run
```

### Specify Output Directory
```bash
python qa_automation.py "Test dashboard" --out cypress/e2e/dashboard
```

## Troubleshooting

### Common Issues
- **Missing OpenAI API key**: Ensure `.env` file exists with `OPENAI_API_KEY=your_key`
- **Import errors**: Run `pip install -r requirements.txt`
- **Cypress not found**: Run `npm install` to install Node dependencies
- **Vector store errors**: Delete `./vector_store` directory and recreate with `--persist-vstore`
- **Poor test quality**: Review and adjust `CY_PROMPT_TEMPLATE` for clearer instructions

### Best Practices
- Start with simple requirements to test the pipeline
- Gradually increase complexity as you validate output quality
- Use the vector store for projects with extensive documentation
- Review generated tests before committing to version control
- Iterate on prompt templates based on actual output quality

## Resources

- [Cypress Documentation](https://docs.cypress.io)
- [LangChain Documentation](https://python.langchain.com)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- [OpenAI API Reference](https://platform.openai.com/docs)
