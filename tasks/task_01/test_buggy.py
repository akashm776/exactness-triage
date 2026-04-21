import pytest
from buggy import process_input

def test_none_value():
    with pytest.raises(ValueError):
        process_input(None, "read")

def test_invalid_mode():
    with pytest.raises(KeyError):
        process_input("data", "delete")

def test_wrong_type():
    with pytest.raises(TypeError):
        process_input(42, "read")
