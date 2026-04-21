from buggy import extract_event

def test_extract():
    record = {
        "timestamp": "2024-01-15T09:00:00",
        "user_id": "u_8821",
        "event_type": "login",
        "priority": 2
    }
    result = extract_event(record)
    assert result["when"] == "2024-01-15T09:00:00"
    assert result["who"] == "u_8821"
    assert result["what"] == "login"
    assert result["severity"] == 2
