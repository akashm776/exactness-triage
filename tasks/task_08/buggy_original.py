def count_down(n):
    return [n] + count_down(n - 1)
