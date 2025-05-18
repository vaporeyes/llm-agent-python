# Agent Guidelines for llm-agent-python

## Build & Test Commands

- Run the agent: `python main.py chat`
- Run with OpenAI: `python main.py chat --provider openai`
- Run with Anthropic: `python main.py chat --provider anthropic`
- Format code: `black .`
- Type checking: `mypy .`

## Code Style Guidelines

- Use type hints for all function parameters and return values
- Follow PEP 8 conventions
- Use black for code formatting
- Class names: PascalCase (e.g., `CodeEditingAgent`)
- Function/method names: snake_case (e.g., `get_provider`)
- Constants: UPPER_SNAKE_CASE
- Use dataclasses for data structures
- Error handling: Use try/except with specific exception types
- Docstrings: Use triple quotes for all classes and methods
- Imports: Group standard library, third-party, and local imports
