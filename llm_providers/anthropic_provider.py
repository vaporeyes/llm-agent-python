"""Anthropic Claude provider implementation."""
from typing import Any, Dict, List, Optional
import anthropic
from plugins.base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""

    def create_client(self, api_key: Optional[str] = None) -> anthropic.Anthropic:
        """Create and return an Anthropic client instance."""
        return anthropic.Anthropic(api_key=api_key)
    
    def run_inference(
        self,
        client: anthropic.Anthropic,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> anthropic.types.Message:
        """Run inference using Claude."""
        return client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            tools=tools if tools else None,
        )
    
    def extract_tool_calls(self, response: anthropic.types.Message) -> List[Dict[str, Any]]:
        """Extract tool calls from Claude's response."""
        tool_calls = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_calls.append({
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": content_block.input,
                })
        return tool_calls
    
    def extract_text_content(self, response: anthropic.types.Message) -> str:
        """Extract text content from Claude's response."""
        text_blocks = []
        for content_block in response.content:
            if content_block.type == "text":
                text_blocks.append(content_block.text)
        return "\n".join(text_blocks) 
    
    def get_token_usage(self, response: anthropic.types.Message) -> Dict[str, int]:
        """Extract token usage information from Claude's response."""
        return {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }