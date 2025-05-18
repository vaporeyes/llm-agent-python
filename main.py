"""
Code-Editing Agent in Python
Based on concepts from: https://ampcode.com/how-to-build-an-agent

This is a Python implementation of the concepts from the blog post about building
a code-editing agent. The agent can read files, list directories, and edit files
using various LLM providers.

Required:
- For Anthropic: pip install anthropic
- For OpenAI: pip install openai
- For Ollama: pip install ollama
"""

import os
import sys
import re
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
import click
from plugins.base import ToolDefinition, LLMProvider
from plugins.file_tools import FileToolsPlugin
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.openai_provider import OpenAIProvider


class CodeEditingAgent:
    """A simple code-editing agent that can interact with files."""

    def __init__(
        self, provider: LLMProvider, plugins: List[Any], api_key: Optional[str] = None,
        show_stats: bool = False, output_file: Optional[str] = None, model_name: str = ""
    ):
        self.provider = provider
        self.client = provider.create_client(api_key)
        self.tools: List[ToolDefinition] = []
        self.conversation: List[Dict[str, Any]] = []
        self.show_stats = show_stats
        self.output_file = output_file
        self.model_name = model_name
        self.conversation_id = str(uuid.uuid4())
        self.chat_history: List[Dict[str, Any]] = []
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.total_time = 0.0
        self._load_plugins(plugins)
        self.system_message = self._get_system_message()

    def _load_plugins(self, plugins: List[Any]) -> None:
        for plugin in plugins:
            self.tools.extend(plugin.get_tools())

    def _get_system_message(self) -> str:
        """Get the system message to guide the model's behavior."""
        return """You are an AI assistant that can help with both general questions and code-related tasks.

When the user asks a general question (like "Why is the sky blue?" or "What is the capital of France?"), 
answer directly without using any tools. These are questions that don't require file operations or code editing.

When the user asks about code, files, or needs to perform file operations, use the available tools to help them.
Examples of code-related tasks:
- Reading or editing files
- Searching through code
- Listing directories
- Making code changes

You should proactively decide when to use tools based on the nature of the query. Use your best judgment to determine when tools would provide a better response than your general knowledge.

For example:
- If a user asks "What's in the current directory?", use the list_files tool without being explicitly asked
- If a user asks about the contents of a specific file, use the read_file tool
- If a user wants to modify code, use the edit_file tool

Always determine the type of question first, then respond appropriately."""

    def run(self) -> None:
        """Main loop for the agent."""
        print("Chat with the agent (use Ctrl+C to quit)")
        print("The agent can help with both general questions and code-related tasks.")

        while True:
            try:
                print("\033[94mYou\033[0m: ", end="", flush=True)
                user_input = input().strip()

                if not user_input:
                    continue

                # Add to internal conversation (for LLM context)
                self.conversation.append({"role": "user", "content": user_input})
                
                # Add to chat history (for export)
                if self.output_file:
                    self.chat_history.append({
                        "id": str(uuid.uuid4()),
                        "conversation_id": self.conversation_id,
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model_name,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0
                        }
                    })

                while True:
                    message = self._run_inference()
                    
                    # Extract text content for display
                    text_content = self.provider.extract_text_content(message)
                    if text_content:
                        print(f"\033[93mAgent\033[0m: {text_content}")
                        # Add the text content to conversation history
                        self.conversation.append({"role": "assistant", "content": text_content})
                        
                        # Add to chat history (for export)
                        if self.output_file:
                            usage = self.provider.get_token_usage(message)
                            self.chat_history.append({
                                "id": str(uuid.uuid4()),
                                "conversation_id": self.conversation_id,
                                "role": "assistant",
                                "content": text_content,
                                "timestamp": datetime.now().isoformat(),
                                "model": self.model_name,
                                "usage": {
                                    "input_tokens": usage["prompt_tokens"],
                                    "output_tokens": usage["completion_tokens"]
                                }
                            })
                            # Save chat history to file after each response
                            self._save_chat_history()

                    # Handle tool calls
                    tool_calls = self.provider.extract_tool_calls(message)
                    if not tool_calls:
                        break

                    # Add tool calls to conversation history
                    tool_call_content = []
                    for tool_call in tool_calls:
                        tool_call_content.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["input"],
                        })
                    self.conversation.append({"role": "assistant", "content": tool_call_content})

                    # Execute tools and collect results
                    tool_results = []
                    for tool_call in tool_calls:
                        result = self._execute_tool(
                            tool_call["id"],
                            tool_call["name"],
                            tool_call["input"],
                        )
                        tool_results.append(result)

                    # Add tool results to conversation history
                    self.conversation.append({"role": "user", "content": tool_results})

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _run_inference(self) -> Any:
        """Send the conversation to the LLM and get a response."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self.tools
        ]

        start_time = time.time()
        response = self.provider.run_inference(
            self.client,
            self.conversation,
            system=self.system_message,
            tools=tools if tools else None,
        )
        elapsed_time = time.time() - start_time
        
        if self.show_stats:
            usage = self.provider.get_token_usage(response)
            self.total_tokens["prompt_tokens"] += usage["prompt_tokens"]
            self.total_tokens["completion_tokens"] += usage["completion_tokens"]
            self.total_tokens["total_tokens"] += usage["total_tokens"]
            self.total_time += elapsed_time
            
            print(f"\033[95mStats\033[0m: Time: {elapsed_time:.2f}s, "
                  f"Tokens: {usage['total_tokens']} "
                  f"(In: {usage['prompt_tokens']}, Out: {usage['completion_tokens']})")
            
        return response

    def _save_chat_history(self) -> None:
        """Save the chat history to the output file."""
        if not self.output_file:
            return
            
        try:
            # Load existing history if file exists
            existing_history = []
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_history = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, start fresh
                    existing_history = []
            
            # Combine existing history with new entries
            combined_history = existing_history + self.chat_history
            
            # Write updated history
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_history, f, indent=2)
                
            # Clear chat history as it's been saved
            self.chat_history = []
        except Exception as e:
            print(f"\033[91mError saving chat history: {e}\033[0m")
    
    def _execute_tool(
        self, tool_id: str, tool_name: str, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        tool_def = next((t for t in self.tools if t.name == tool_name), None)

        if not tool_def:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "Tool not found",
                "is_error": True,
            }

        print(f"\033[92mTool\033[0m: {tool_name}({tool_input})")

        try:
            result = tool_def.function(tool_input)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result,
                "is_error": False,
            }
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(e),
                "is_error": True,
            }


def get_provider(provider_name: str) -> tuple[LLMProvider, str]:
    """Get the appropriate LLM provider and API key."""
    provider_name = provider_name.lower()

    if provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.ClickException(
                "Please set the ANTHROPIC_API_KEY environment variable"
            )
        return AnthropicProvider(), api_key
    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise click.ClickException(
                "Please set the OPENAI_API_KEY environment variable"
            )
        return OpenAIProvider(), api_key
    else:
        raise click.ClickException(
            f"Unsupported LLM provider: {provider_name}\n"
            "Supported providers: anthropic, openai"
        )


@click.group()
def cli():
    """LLM Agent CLI - A code-editing agent that can interact with files."""
    pass


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="The LLM provider to use",
)
@click.option(
    "--model",
    "-m",
    help="The specific model to use (provider-specific)",
)
@click.option(
    "--stats",
    is_flag=True,
    help="Show token usage and timing statistics",
)
@click.option(
    "--output",
    "-o",
    help="Save chat history to the specified file (default: chat_history.json)",
    default=None,
)
def chat(provider: str, model: Optional[str], stats: bool, output: Optional[str]):
    """Start an interactive chat session with the agent."""
    try:
        provider_instance, api_key = get_provider(provider)
        output_file = output if output else ('chat_history.json' if output is not None else None)
        model_name = model or ("claude-3-7-sonnet-latest" if provider == "anthropic" else "gpt-4-turbo-preview")
        
        plugins = [FileToolsPlugin()]
        agent = CodeEditingAgent(
            provider_instance, 
            plugins, 
            api_key, 
            show_stats=stats,
            output_file=output_file,
            model_name=f"{provider}/{model_name}"
        )
        agent.run()
        
        if stats:
            # Print final statistics
            print("\n" + "-" * 50)
            print(f"\033[95mFinal Stats\033[0m:")
            print(f"Total time: {agent.total_time:.2f}s")
            print(f"Total tokens: {agent.total_tokens['total_tokens']}")
            print(f"  - Input tokens: {agent.total_tokens['prompt_tokens']}")
            print(f"  - Output tokens: {agent.total_tokens['completion_tokens']}")
            print("-" * 50)
    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    cli()
