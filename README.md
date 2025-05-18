# LLM Agent Python

A Python implementation of a code-editing agent that can help with both general questions and code-related tasks. The agent uses various LLM providers (Anthropic Claude, OpenAI) and can read files, list directories, and edit files.

## Features

- Interactive chat interface with multiple LLM provider support
- File operations (read, list, edit)
- Tool usage statistics tracking
- Chat history export
- Proactive tool selection based on query type

## Installation

This project uses `uv` for Python package management.

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or install directly from pyproject.toml
uv pip install -e .
```

## Usage

```bash
# Set your API keys as environment variables
export ANTHROPIC_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"

# Start a chat session with default provider (Anthropic)
python main.py chat

# Use OpenAI instead
python main.py chat --provider openai
```

### CLI Options

```text
Options:
  -p, --provider [anthropic|openai]  The LLM provider to use
  -m, --model TEXT                   The specific model to use (provider-specific)
  --stats                            Show token usage and timing statistics
  -o, --output TEXT                  Save chat history to the specified file (default: chat_history.json)
  --help                             Show this message and exit
```

## Examples

```bash
# Chat with Claude and show stats
python main.py chat --provider anthropic --stats

# Chat with OpenAI GPT-4 and export chat history
python main.py chat --provider openai --output my_chat.json

# Combine stats and chat history export
python main.py chat --stats --output
```

## Project Structure

```text
llm-agent-python/
├── llm_providers/         # LLM provider implementations
│   ├── anthropic_provider.py
│   └── openai_provider.py
├── plugins/               # Tool plugins
│   ├── base.py            # Base classes and interfaces
│   └── file_tools.py      # File operation tools
├── tests/                 # Unit tests
│   ├── llm_providers/     # Tests for LLM providers
│   ├── plugins/           # Tests for plugins
│   └── test_main.py       # Tests for main functionality
├── main.py                # Main CLI application
├── pyproject.toml         # Project dependencies
└── chat_history.json      # Default chat history output file
```

## Adding New Providers

To add a new LLM provider:

1. Create a new file in `llm_providers/` directory
2. Implement the `LLMProvider` interface from `plugins/base.py`
3. Add provider selection in the `get_provider()` function in `main.py`

## Adding New Tools

To add new tools:

1. Create a new plugin class that extends `BasePlugin`
2. Implement the `get_tools()` method to return a list of `ToolDefinition` objects
3. Add the plugin to the `plugins` list in the `chat()` function

## Testing

The project uses pytest for unit testing. Comprehensive tests have been written for all components:

- LLM Provider implementations
- Tool plugins
- The main CodeEditingAgent class
- CLI functionality

```bash
# Install test dependencies
uv sync

# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_main.py

# Run with verbose output
uv run pytest -v

# Run with code coverage
uv run pytest --cov=.
```
