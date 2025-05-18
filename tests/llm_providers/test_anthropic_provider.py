import pytest
from unittest.mock import MagicMock, patch
from llm_providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def provider():
    """Create an AnthropicProvider instance."""
    return AnthropicProvider()


def test_create_client(provider):
    """Test client creation."""
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = provider.create_client("test-api-key")
        mock_anthropic.assert_called_once_with(api_key="test-api-key")


def test_run_inference(provider):
    """Test running inference."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_client.messages.create.return_value = mock_response
    
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"name": "test_tool", "description": "A test tool", "input_schema": {}}]
    
    result = provider.run_inference(
        mock_client,
        messages,
        tools=tools,
        system="You are a helpful assistant.",
        max_tokens=100
    )
    
    mock_client.messages.create.assert_called_once_with(
        model="claude-3-7-sonnet-latest",
        max_tokens=100,
        messages=messages,
        system="You are a helpful assistant.",
        tools=tools
    )
    assert result == mock_response


def test_extract_tool_calls(provider):
    """Test extracting tool calls from response."""
    mock_response = MagicMock()
    mock_tool_use = MagicMock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.id = "tool1"
    mock_tool_use.name = "test_tool"
    mock_tool_use.input = {"param": "value"}
    
    mock_text = MagicMock()
    mock_text.type = "text"
    mock_text.text = "Some text response"
    
    mock_response.content = [mock_text, mock_tool_use]
    
    tool_calls = provider.extract_tool_calls(mock_response)
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tool1"
    assert tool_calls[0]["name"] == "test_tool"
    assert tool_calls[0]["input"] == {"param": "value"}


def test_extract_text_content(provider):
    """Test extracting text content from response."""
    mock_response = MagicMock()
    mock_content = [
        MagicMock(type="text", text="First text block"),
        MagicMock(type="tool_use", id="tool1", name="test_tool", input={}),
        MagicMock(type="text", text="Second text block")
    ]
    mock_response.content = mock_content
    
    text = provider.extract_text_content(mock_response)
    
    assert "First text block" in text
    assert "Second text block" in text


def test_get_token_usage(provider):
    """Test extracting token usage from response."""
    mock_response = MagicMock()
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 100
    
    usage = provider.get_token_usage(mock_response)
    
    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 100
    assert usage["total_tokens"] == 150