import pytest
from unittest.mock import MagicMock, patch
from llm_providers.openai_provider import OpenAIProvider


@pytest.fixture
def provider():
    """Create an OpenAIProvider instance."""
    return OpenAIProvider()


def test_create_client(provider):
    """Test client creation."""
    with patch("llm_providers.openai_provider.OpenAI") as mock_openai:
        client = provider.create_client("test-api-key")
        mock_openai.assert_called_once_with(api_key="test-api-key")


def test_run_inference(provider):
    """Test running inference with system message."""
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat
    
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"name": "test_tool", "description": "A test tool", "input_schema": {}}]
    
    result = provider.run_inference(
        mock_client,
        messages,
        tools=tools,
        system="You are a helpful assistant.",
        max_tokens=100
    )
    
    # Check that system message was added
    expected_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo-preview",
        max_tokens=100,
        messages=expected_messages,
        tools=tools
    )
    assert result == mock_chat


def test_run_inference_no_system(provider):
    """Test running inference without system message."""
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_client.chat.completions.create.return_value = mock_chat
    
    messages = [{"role": "user", "content": "Hello"}]
    
    result = provider.run_inference(
        mock_client,
        messages,
        system=None
    )
    
    # Check that messages were not modified
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4-turbo-preview",
        max_tokens=1024,  # Default
        messages=messages,
        tools=None
    )


def test_extract_tool_calls(provider):
    """Test extracting tool calls from response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_response.choices = [mock_choice]
    
    # Create a message with tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "tool1"
    mock_tool_call.function.name = "test_tool"
    mock_tool_call.function.arguments = '{"param": "value"}'
    
    mock_choice.message.tool_calls = [mock_tool_call]
    
    tool_calls = provider.extract_tool_calls(mock_response)
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tool1"
    assert tool_calls[0]["name"] == "test_tool"
    assert tool_calls[0]["input"] == '{"param": "value"}'


def test_extract_tool_calls_no_tools(provider):
    """Test extracting tool calls when there are none."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_choice.message.tool_calls = None
    
    tool_calls = provider.extract_tool_calls(mock_response)
    
    assert len(tool_calls) == 0


def test_extract_text_content(provider):
    """Test extracting text content from response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_response.choices = [mock_choice]
    
    mock_choice.message.content = "This is a test response"
    
    text = provider.extract_text_content(mock_response)
    
    assert text == "This is a test response"


def test_extract_text_content_multiple_choices(provider):
    """Test extracting text content from multiple choices."""
    mock_response = MagicMock()
    mock_choice1 = MagicMock()
    mock_choice2 = MagicMock()
    mock_response.choices = [mock_choice1, mock_choice2]
    
    mock_choice1.message.content = "First response"
    mock_choice2.message.content = "Second response"
    
    text = provider.extract_text_content(mock_response)
    
    assert "First response" in text
    assert "Second response" in text


def test_get_token_usage(provider):
    """Test extracting token usage from response."""
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 100
    mock_response.usage.total_tokens = 150
    
    usage = provider.get_token_usage(mock_response)
    
    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 100
    assert usage["total_tokens"] == 150