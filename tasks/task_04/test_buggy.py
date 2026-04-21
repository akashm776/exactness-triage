from buggy import format_timestamp

def test_standard():
    assert format_timestamp(2024, 3, 15, 9, 5) == "2024-03-15 09:05"

def test_padding():
    assert format_timestamp(2024, 11, 2, 14, 30) == "2024-11-02 14:30"
