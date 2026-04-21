from buggy import running_product

def test_basic():
    assert running_product([2, 3, 4]) == 24

def test_single():
    assert running_product([5]) == 5
