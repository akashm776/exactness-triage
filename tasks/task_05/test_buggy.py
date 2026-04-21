from buggy import is_valid_score

def test_valid():
    assert is_valid_score(50) == True
    assert is_valid_score(0) == True
    assert is_valid_score(100) == True

def test_invalid():
    assert is_valid_score(-1) == False
    assert is_valid_score(101) == False
