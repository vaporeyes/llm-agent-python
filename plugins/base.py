from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    function: Callable[[Dict[str, Any]], str]

class BasePlugin(ABC):
    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        pass

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def create_client(self, api_key: Optional[str] = None) -> Any:
        """Create and return a client instance for the LLM provider."""
        pass
    
    @abstractmethod
    def run_inference(
        self,
        client: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Any:
        """Run inference using the LLM provider."""
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from the LLM response."""
        pass
    
    @abstractmethod
    def extract_text_content(self, response: Any) -> str:
        """Extract text content from the LLM response."""
        pass
    
    @abstractmethod
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage information from the LLM response.
        
        Returns a dictionary with keys 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
        """
        pass 