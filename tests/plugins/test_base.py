import pytest
from plugins.base import ToolDefinition, BasePlugin, LLMProvider
from typing import Any, Dict, List, Optional


def test_tool_definition():
    """Test that a ToolDefinition can be created properly."""
    def dummy_function(input_data: Dict[str, Any]) -> str:
        return f"Processed: {input_data}"
    
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"param": {"type": "string"}}},
        function=dummy_function
    )
    
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert "type" in tool.input_schema
    assert callable(tool.function)
    
    # Test function execution
    result = tool.function({"param": "test value"})
    assert "Processed:" in result
    assert "test value" in str(result)


class TestBasePlugin(BasePlugin):
    """Concrete implementation for testing BasePlugin."""
    def get_tools(self) -> List[ToolDefinition]:
        def dummy_func(input_data: Dict[str, Any]) -> str:
            return "Test"
        
        return [
            ToolDefinition(
                name="dummy_tool",
                description="A dummy tool",
                input_schema={},
                function=dummy_func
            )
        ]


def test_base_plugin():
    """Test BasePlugin implementation."""
    plugin = TestBasePlugin()
    tools = plugin.get_tools()
    
    assert len(tools) == 1
    assert tools[0].name == "dummy_tool"
    assert tools[0].function({}) == "Test"