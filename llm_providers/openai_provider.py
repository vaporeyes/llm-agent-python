"""OpenAI provider implementation."""
from typing import Any, Dict, List, Optional
from openai import OpenAI
from plugins.base import LLMProvider

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI's models."""
    
    def create_client(self, api_key: Optional[str] = None) -> OpenAI:
        """Create and return an OpenAI client instance."""
        return OpenAI(api_key=api_key)
    
    def run_inference(
        self,
        client: OpenAI,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Any:
        """Run inference using OpenAI."""
        # Add system message to the beginning of the messages list
        if system:
            messages = [{"role": "system", "content": system}] + messages
            
        return client.chat.completions.create(
            model="gpt-4-turbo-preview",
            max_tokens=max_tokens,
            messages=messages,
            tools=tools if tools else None,
        )
    
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI's response."""
        tool_calls = []
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": tool_call.function.arguments,
                    })
        return tool_calls
    
    def extract_text_content(self, response: Any) -> str:
        """Extract text content from OpenAI's response."""
        text_blocks = []
        for choice in response.choices:
            if choice.message.content:
                text_blocks.append(choice.message.content)
        return "\n".join(text_blocks) 
    
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from OpenAI's response."""
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }