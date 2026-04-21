from buggy import get_middle_elements

def test_five():
    assert get_middle_elements([1, 2, 3, 4, 5]) == [2, 3, 4]

def test_three():
    assert get_middle_elements([1, 2, 3]) == [2]

def test_four():
    assert get_middle_elements([1, 2, 3, 4]) == [2, 3]
