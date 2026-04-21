from buggy import count_down

def test_basic():
    assert count_down(3) == [3, 2, 1]

def test_one():
    assert count_down(1) == [1]
