import pytest
import os
import json
from unittest.mock import MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

from main import CodeEditingAgent, get_provider
from plugins.base import LLMProvider, ToolDefinition
from plugins.file_tools import FileToolsPlugin


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.client = MagicMock()
        self.response = MagicMock()
        self.text_content = "This is a test response"
        self.tool_calls = []
    
    def create_client(self, api_key=None):
        return self.client
    
    def run_inference(self, client, messages, tools=None, system=None, max_tokens=1024):
        return self.response
    
    def extract_tool_calls(self, response):
        return self.tool_calls
    
    def extract_text_content(self, response):
        return self.text_content
    
    def get_token_usage(self, response):
        return {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def agent(mock_provider):
    """Create a CodeEditingAgent with a mock provider."""
    plugins = [FileToolsPlugin()]
    return CodeEditingAgent(mock_provider, plugins, "test-api-key")


def test_agent_initialization(mock_provider):
    """Test agent initialization."""
    plugins = [FileToolsPlugin()]
    agent = CodeEditingAgent(
        mock_provider, 
        plugins, 
        "test-api-key", 
        show_stats=True,
        output_file="test_output.json",
        model_name="test/model"
    )
    
    assert agent.provider == mock_provider
    assert len(agent.tools) > 0  # Should have loaded tools from the plugin
    assert agent.show_stats is True
    assert agent.output_file == "test_output.json"
    assert agent.model_name == "test/model"
    assert agent.conversation_id != ""
    assert agent.total_tokens["prompt_tokens"] == 0
    assert agent.total_tokens["completion_tokens"] == 0
    assert agent.total_tokens["total_tokens"] == 0


def test_run_inference(agent, mock_provider):
    """Test running inference."""
    # Set up the mock to return specific token usage
    mock_provider.response.usage = MagicMock()
    mock_provider.response.usage.prompt_tokens = 10
    mock_provider.response.usage.completion_tokens = 20
    
    # Run inference with stats enabled
    agent.show_stats = True
    response = agent._run_inference()
    
    # Verify token usage was updated
    assert agent.total_tokens["prompt_tokens"] == 10
    assert agent.total_tokens["completion_tokens"] == 20
    assert agent.total_tokens["total_tokens"] == 30
    assert response == mock_provider.response


def test_execute_tool(agent):
    """Test tool execution."""
    # Create a simple test tool
    def test_func(input_data):
        return f"Result: {input_data['value']}"
    
    agent.tools = [
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={},
            function=test_func
        )
    ]
    
    # Execute the tool
    result = agent._execute_tool(
        "tool123",
        "test_tool",
        {"value": "test"}
    )
    
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tool123"
    assert result["content"] == "Result: test"
    assert result["is_error"] is False


def test_execute_tool_not_found(agent):
    """Test executing a non-existent tool."""
    result = agent._execute_tool(
        "tool123",
        "non_existent_tool",
        {}
    )
    
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tool123"
    assert "Tool not found" in result["content"]
    assert result["is_error"] is True


def test_execute_tool_error(agent):
    """Test tool execution with an error."""
    # Create a tool that raises an exception
    def failing_func(input_data):
        raise ValueError("Test error")
    
    agent.tools = [
        ToolDefinition(
            name="failing_tool",
            description="A failing tool",
            input_schema={},
            function=failing_func
        )
    ]
    
    # Execute the tool
    result = agent._execute_tool(
        "tool123",
        "failing_tool",
        {}
    )
    
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "tool123"
    assert "Test error" in result["content"]
    assert result["is_error"] is True


@pytest.fixture
def temp_output_file(tmp_path):
    """Create a temporary output file."""
    output_file = tmp_path / "test_chat_history.json"
    yield str(output_file)
    # Clean up
    if output_file.exists():
        output_file.unlink()


def test_save_chat_history(mock_provider, temp_output_file):
    """Test saving chat history to file."""
    # Create agent with output file
    plugins = [FileToolsPlugin()]
    agent = CodeEditingAgent(
        mock_provider, 
        plugins, 
        "test-api-key", 
        output_file=temp_output_file,
        model_name="test/model"
    )
    
    # Add some chat history
    agent.chat_history = [
        {
            "id": "msg1",
            "conversation_id": agent.conversation_id,
            "role": "user",
            "content": "Hello",
            "timestamp": datetime.now().isoformat(),
            "model": "test/model",
            "usage": {"input_tokens": 0, "output_tokens": 0}
        },
        {
            "id": "msg2",
            "conversation_id": agent.conversation_id,
            "role": "assistant",
            "content": "Hi there",
            "timestamp": datetime.now().isoformat(),
            "model": "test/model",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
    ]
    
    # Save the chat history
    agent._save_chat_history()
    
    # Verify the file was created and contains the chat history
    assert os.path.exists(temp_output_file)
    
    with open(temp_output_file, 'r') as f:
        saved_history = json.load(f)
    
    assert len(saved_history) == 2
    assert saved_history[0]["id"] == "msg1"
    assert saved_history[0]["role"] == "user"
    assert saved_history[0]["content"] == "Hello"
    assert saved_history[1]["id"] == "msg2"
    assert saved_history[1]["role"] == "assistant"
    assert saved_history[1]["content"] == "Hi there"
    
    # Chat history should be cleared after saving
    assert len(agent.chat_history) == 0


def test_save_chat_history_append(mock_provider, temp_output_file):
    """Test appending to existing chat history file."""
    # Create a file with existing history
    existing_history = [
        {
            "id": "old_msg",
            "conversation_id": "old_convo",
            "role": "user",
            "content": "Previous message",
            "timestamp": datetime.now().isoformat(),
            "model": "test/model",
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    ]
    
    with open(temp_output_file, 'w') as f:
        json.dump(existing_history, f)
    
    # Create agent with output file
    plugins = [FileToolsPlugin()]
    agent = CodeEditingAgent(
        mock_provider, 
        plugins, 
        "test-api-key", 
        output_file=temp_output_file,
        model_name="test/model"
    )
    
    # Add new chat history
    agent.chat_history = [
        {
            "id": "new_msg",
            "conversation_id": agent.conversation_id,
            "role": "user",
            "content": "New message",
            "timestamp": datetime.now().isoformat(),
            "model": "test/model",
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    ]
    
    # Save the chat history
    agent._save_chat_history()
    
    # Verify the file contains both old and new history
    with open(temp_output_file, 'r') as f:
        saved_history = json.load(f)
    
    assert len(saved_history) == 2
    assert saved_history[0]["id"] == "old_msg"
    assert saved_history[0]["content"] == "Previous message"
    assert saved_history[1]["id"] == "new_msg"
    assert saved_history[1]["content"] == "New message"


@patch('os.getenv')
def test_get_provider_anthropic(mock_getenv):
    """Test getting Anthropic provider."""
    mock_getenv.return_value = "test-api-key"
    
    provider, api_key = get_provider("anthropic")
    
    assert api_key == "test-api-key"
    from llm_providers.anthropic_provider import AnthropicProvider
    assert isinstance(provider, AnthropicProvider)


@patch('os.getenv')
def test_get_provider_openai(mock_getenv):
    """Test getting OpenAI provider."""
    mock_getenv.return_value = "test-api-key"
    
    provider, api_key = get_provider("openai")
    
    assert api_key == "test-api-key"
    from llm_providers.openai_provider import OpenAIProvider
    assert isinstance(provider, OpenAIProvider)


@patch('os.getenv')
def test_get_provider_error(mock_getenv):
    """Test getting an unsupported provider."""
    mock_getenv.return_value = "test-api-key"
    
    with pytest.raises(Exception) as exc_info:
        provider, api_key = get_provider("unsupported")
    
    assert "Unsupported LLM provider" in str(exc_info.value)