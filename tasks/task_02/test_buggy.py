from buggy import serialize_record

def test_basic():
    assert serialize_record(["alice", "30", "eng"]) == "alice::30::eng"

def test_single():
    assert serialize_record(["bob"]) == "bob"

def test_empty_field():
    assert serialize_record(["x", "", "z"]) == "x::z"
