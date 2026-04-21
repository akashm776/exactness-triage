def extract_event(record):
    return {
        "when": record["ts"],
        "who": record["uid"],
        "what": record["evt"],
        "severity": record["lvl"]
    }
