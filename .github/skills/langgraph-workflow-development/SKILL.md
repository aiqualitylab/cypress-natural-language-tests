---
name: langgraph-workflow-development
description: Guide for LangGraph workflows in AI-powered test automation
LangGraph workflows in AI-powered test automation license: GNU AFFERO GENERAL PUBLIC LICENSE
---

# LangGraph Workflow Development Skill

## Workflow Pipeline

```
ParseCLI -> LoadContext -> GenerateTests -> RunCypress -> END
```

## State Schema

```python
@dataclass
class TestGenerationState:
    requirements: List[str]
    output_dir: str
    use_prompt: bool
    docs_context: Optional[str]
    generated_tests: List[Dict[str, Any]]
    run_tests: bool
    error: Optional[str]
```

## Building Workflow

```python
from langgraph.graph import StateGraph, END

def create_workflow() -> StateGraph:
    workflow = StateGraph(TestGenerationState)
    
    workflow.add_node("parse_cli", parse_cli_node)
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("generate_tests", generate_tests_node)
    workflow.add_node("run_cypress", run_cypress_node)
    
    workflow.set_entry_point("parse_cli")
    workflow.add_edge("parse_cli", "load_context")
    workflow.add_edge("load_context", "generate_tests")
    workflow.add_edge("generate_tests", "run_cypress")
    workflow.add_edge("run_cypress", END)
    
    return workflow.compile()
```

## Node Pattern

```python
def generate_tests_node(state: TestGenerationState) -> TestGenerationState:
    generator = HybridTestGenerator()
    
    for idx, req in enumerate(state.requirements, 1):
        content = generator.generate_test_content(
            requirement=req,
            context=state.docs_context or "",
            use_prompt=state.use_prompt
        )
        result = generator.save_test_file(content, state.output_dir, idx)
        state.generated_tests.append(result)
    
    return state
```

## Mode Selection

```python
# Template
if state.use_prompt:
    template = create_prompt_powered_test_prompt()
else:
    template = create_traditional_test_prompt()

# Output folder
if state.use_prompt:
    folder = f"{output_dir}/prompt-powered"
else:
    folder = f"{output_dir}/generated"
```

## Context Sources

```python
# URL analysis
if args.url:
    context, data, path = generate_test_data_from_url(args.url)

# JSON file
if args.data:
    with open(args.data) as f:
        data = json.load(f)

# Documentation
if args.docs:
    docs_context = loader.get_relevant_context(vector_store, query)

# Combine all into docs_context
full_context = (docs_context or "") + test_data_context
```

## Error Handling

```python
def node_with_error_handling(state):
    try:
        result = process(state)
        return result
    except Exception as e:
        state.error = str(e)
        return state
```