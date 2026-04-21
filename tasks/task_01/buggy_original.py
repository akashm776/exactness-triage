def process_input(value, mode):
    if value is None:
        raise RuntimeError("value required")
    if mode not in ("read", "write", "append"):
        raise RuntimeError("invalid mode")
    if not isinstance(value, str):
        raise RuntimeError("value must be string")
    return value.strip()
