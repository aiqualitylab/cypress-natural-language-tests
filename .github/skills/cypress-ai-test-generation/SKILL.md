---
name: cypress-ai-test-generation
description: Guide for AI-powered Cypress test generation from natural language
Cypress test generation from natural language license: GNU AFFERO GENERAL PUBLIC LICENSE
---

# Cypress AI Test Generation Skill

Generate Cypress E2E tests from natural language using OpenAI GPT-4o-mini.

## CLI Quick Reference

| Flag | Purpose |
|------|---------|
| `--url`, `-u` | Fetch URL, analyze HTML, generate fixture |
| `--data`, `-d` | Load JSON test data file |
| `--use-prompt` | Generate cy.prompt() self-healing tests |
| `--run` | Execute tests after generation |
| `--analyze`, `-a` | Diagnose test failure with AI |

## Two Test Modes

**Traditional** (`cypress/e2e/generated/`)
- Uses fixture data, dynamic selectors
- Best for CI/CD

**cy.prompt()** (`cypress/e2e/prompt-powered/`)
- Self-healing tests
- Best for development

## Dynamic Test Pattern

```javascript
describe('Tests', function () {
    beforeEach(function () {
        cy.fixture('url_test_data').then((data) => {
            this.testData = data;
        });
    });

    it('test case', function () {
        cy.visit(this.testData.url);
        const valid = this.testData.test_cases.find(tc => tc.name === 'valid_test');
        const selectors = this.testData.selectors;
        
        Object.keys(selectors).forEach(field => {
            if (field !== 'submit' && valid[field]) {
                cy.get(selectors[field]).type(valid[field]);
            }
        });
        
        cy.get(selectors.submit).click();
    });
});
```

## Fixture Structure

```json
{
  "url": "https://example.com",
  "selectors": {"field": "#selector", "submit": "button"},
  "test_cases": [
    {"name": "valid_test", "field": "value"},
    {"name": "invalid_test", "field": "wrong"}
  ]
}
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `this.testData` undefined | Use `function()` not `=>` |
| Wrong selectors | Use `--url` to fetch real selectors |
| Tests only work for one URL | Use dynamic selector pattern |

## File Organization

```
cypress/
├── e2e/
│   ├── generated/
│   └── prompt-powered/
└── fixtures/
    └── url_test_data.json
```