import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BasePlugin, ToolDefinition

class FileToolsPlugin(BasePlugin):
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="read_file",
                description="Read the contents of a given relative file path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The relative path of a file in the working directory.",
                        }
                    },
                    "required": ["path"],
                },
                function=self._read_file,
            ),
            ToolDefinition(
                name="list_files",
                description="List files and directories at a given path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional relative path to list files from.",
                        }
                    },
                    "required": [],
                },
                function=self._list_files,
            ),
            ToolDefinition(
                name="edit_file",
                description="Make edits to a text file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file"},
                        "old_str": {
                            "type": "string",
                            "description": "Text to search for",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Text to replace old_str with",
                        },
                    },
                    "required": ["path", "old_str", "new_str"],
                },
                function=self._edit_file,
            ),
        ]

    def _read_file(self, input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", "")
        if not path:
            raise ValueError("Path is required")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading file {path}: {e}")

    def _list_files(self, input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", ".")
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                raise Exception(f"Path {path} does not exist")
            files = []
            directories = []
            for item in path_obj.iterdir():
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    directories.append(item.name + "/")
            return json.dumps(sorted(directories) + sorted(files))
        except Exception as e:
            raise Exception(f"Error listing files in {path}: {e}")

    def _edit_file(self, input_data: Dict[str, Any]) -> str:
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")
        if not path:
            raise ValueError("Path is required")
        if old_str == new_str:
            raise ValueError("old_str and new_str must be different")
        path_obj = Path(path)
        if not path_obj.exists() and old_str == "":
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_str)
            return f"Successfully created file {path}"
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Error reading file {path}: {e}")
        if old_str == "":
            new_content = content + new_str
        else:
            if old_str not in content:
                raise Exception(f"old_str not found in file {path}")
            new_content = content.replace(old_str, new_str)
        try:
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_content)
            return "OK"
        except Exception as e:
            raise Exception(f"Error writing to file {path}: {e}") 