import pytest
import os
import tempfile
import json
from typing import Dict, Any
from plugins.file_tools import FileToolsPlugin


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def plugin():
    """Create a FileToolsPlugin instance."""
    return FileToolsPlugin()


@pytest.fixture
def test_file(temp_dir):
    """Create a test file with content."""
    file_path = os.path.join(temp_dir, "test_file.txt")
    content = "This is a test file.\nIt has multiple lines.\nFor testing."
    
    with open(file_path, "w") as f:
        f.write(content)
    
    return file_path


def test_read_file(plugin, test_file):
    """Test the read_file tool."""
    result = plugin._read_file({"path": test_file})
    
    assert "This is a test file." in result
    assert "It has multiple lines." in result
    assert "For testing." in result


def test_read_file_error(plugin):
    """Test read_file with an invalid path."""
    with pytest.raises(Exception) as exc_info:
        plugin._read_file({"path": "/nonexistent/path"})
    
    assert "Error reading file" in str(exc_info.value)


def test_list_files(plugin, temp_dir, test_file):
    """Test the list_files tool."""
    # Create another file and a directory for testing
    os.makedirs(os.path.join(temp_dir, "subdir"))
    with open(os.path.join(temp_dir, "another.txt"), "w") as f:
        f.write("Another test file")
    
    result = plugin._list_files({"path": temp_dir})
    result_data = json.loads(result)
    
    assert "test_file.txt" in result_data
    assert "another.txt" in result_data
    assert "subdir/" in result_data


def test_list_files_error(plugin):
    """Test list_files with an invalid path."""
    with pytest.raises(Exception) as exc_info:
        plugin._list_files({"path": "/nonexistent/path"})
    
    assert "Error listing files" in str(exc_info.value)


def test_edit_file(plugin, test_file):
    """Test the edit_file tool."""
    # Replace text in the file
    result = plugin._edit_file({
        "path": test_file,
        "old_str": "This is a test file.",
        "new_str": "This is a modified test file."
    })
    
    assert result == "OK"
    
    # Verify the file was edited
    with open(test_file, "r") as f:
        content = f.read()
    
    assert "This is a modified test file." in content
    assert "This is a test file." not in content


def test_edit_file_create(plugin, temp_dir):
    """Test edit_file creating a new file."""
    new_file = os.path.join(temp_dir, "new_file.txt")
    
    result = plugin._edit_file({
        "path": new_file,
        "old_str": "",
        "new_str": "This is a new file."
    })
    
    assert "Successfully created file" in result
    assert os.path.exists(new_file)
    
    with open(new_file, "r") as f:
        content = f.read()
    
    assert "This is a new file." in content


def test_edit_file_error(plugin, test_file):
    """Test edit_file with errors."""
    # Test with non-existent string
    with pytest.raises(Exception) as exc_info:
        plugin._edit_file({
            "path": test_file,
            "old_str": "This string doesn't exist",
            "new_str": "New text"
        })
    
    assert "old_str not found" in str(exc_info.value)
    
    # Test with same old and new strings
    with pytest.raises(ValueError) as exc_info:
        plugin._edit_file({
            "path": test_file,
            "old_str": "This is a test file.",
            "new_str": "This is a test file."
        })
    
    assert "old_str and new_str must be different" in str(exc_info.value)