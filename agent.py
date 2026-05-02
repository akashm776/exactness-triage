import anthropic
import subprocess
import sqlite3
import re
import uuid
import os
import time
from datetime import datetime

client = anthropic.Anthropic()

# ── Token Volume Proxy ────────────────────────────────────

def approx_token_volume(text: str) -> int:
    """
    Approximate token volume using character count proxy.
    Assumes ~4 chars per token. Rough estimate used for
    relative comparison across conditions only.
    Not an exact token count.
    """
    return max(1, len(text) // 4)

# ── Database ──────────────────────────────────────────────

def init_db(db_path="observations.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            task_id TEXT,
            task_type TEXT,
            replicate INTEGER,
            step INTEGER,
            tool_name TEXT,
            raw_content TEXT,
            processed_content TEXT,
            approx_raw_tokens INTEGER,
            approx_processed_tokens INTEGER,
            exactness_label INTEGER,
            exactness_confidence REAL,
            condition TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            task_id TEXT,
            task_type TEXT,
            replicate INTEGER,
            condition TEXT,
            passed INTEGER,
            approx_ctx_token_volume INTEGER,
            total_billed_tokens INTEGER,
            latency_seconds REAL,
            steps INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn

def log_observation(conn, session_id, task_id, task_type,
                    replicate, step, tool_name,
                    raw, processed, exactness,
                    confidence, condition):
    raw_tok = approx_token_volume(raw)
    proc_tok = approx_token_volume(processed)
    conn.execute(
        "INSERT INTO observations VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), session_id, task_id, task_type,
         replicate, step, tool_name, raw, processed,
         raw_tok, proc_tok, int(exactness), confidence,
         condition, datetime.utcnow().isoformat())
    )
    conn.commit()
    return proc_tok

def log_session(conn, session_id, task_id, task_type,
                replicate, condition, passed,
                ctx_tokens, total_billed, latency, steps):
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (session_id, task_id, task_type, replicate,
         condition, int(passed), ctx_tokens,
         total_billed, latency, steps,
         datetime.utcnow().isoformat())
    )
    conn.commit()

# ── Exactness Classifier ──────────────────────────────────

LIKELY_EXACT_TOOLS = {"run_tests"}
LIKELY_NONEXACT_TOOLS = {"write_file"}

# ── Context Pressure Noise Pool ───────────────────────────
# 8 non-exact (verbose, summarisable) + 4 exact (load-bearing specifics).
# Injected as synthetic past tool calls before the agent loop when
# pressure=True, to simulate a session that has already accumulated context.

PRESSURE_NOISE = [
    # ── non-exact: passing test runs ──────────────────────
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_math.py",
                  "code_file": "utils/math_helpers.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 18 items\n\n"
            "utils/test_math.py::test_add PASSED                            [  5%]\n"
            "utils/test_math.py::test_subtract PASSED                       [ 11%]\n"
            "utils/test_math.py::test_multiply PASSED                       [ 16%]\n"
            "utils/test_math.py::test_divide PASSED                         [ 22%]\n"
            "utils/test_math.py::test_divide_by_zero PASSED                 [ 27%]\n"
            "utils/test_math.py::test_modulo PASSED                         [ 33%]\n"
            "utils/test_math.py::test_power PASSED                          [ 38%]\n"
            "utils/test_math.py::test_sqrt PASSED                           [ 44%]\n"
            "utils/test_math.py::test_floor_div PASSED                      [ 50%]\n"
            "utils/test_math.py::test_abs PASSED                            [ 55%]\n"
            "utils/test_math.py::test_round PASSED                          [ 61%]\n"
            "utils/test_math.py::test_clamp PASSED                          [ 66%]\n"
            "utils/test_math.py::test_lerp PASSED                           [ 72%]\n"
            "utils/test_math.py::test_normalize PASSED                      [ 77%]\n"
            "utils/test_math.py::test_sigmoid PASSED                        [ 83%]\n"
            "utils/test_math.py::test_softmax PASSED                        [ 88%]\n"
            "utils/test_math.py::test_log_sum_exp PASSED                    [ 94%]\n"
            "utils/test_math.py::test_batch_norm PASSED                     [100%]\n\n"
            "============================== 18 passed in 0.31s "
            "=============================="
        ),
    },
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_strings.py",
                  "code_file": "utils/string_utils.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 14 items\n\n"
            "utils/test_strings.py::test_strip PASSED                       [  7%]\n"
            "utils/test_strings.py::test_upper PASSED                       [ 14%]\n"
            "utils/test_strings.py::test_lower PASSED                       [ 21%]\n"
            "utils/test_strings.py::test_split PASSED                       [ 28%]\n"
            "utils/test_strings.py::test_join PASSED                        [ 35%]\n"
            "utils/test_strings.py::test_replace PASSED                     [ 42%]\n"
            "utils/test_strings.py::test_startswith PASSED                  [ 50%]\n"
            "utils/test_strings.py::test_endswith PASSED                    [ 57%]\n"
            "utils/test_strings.py::test_contains PASSED                    [ 64%]\n"
            "utils/test_strings.py::test_truncate PASSED                    [ 71%]\n"
            "utils/test_strings.py::test_pad PASSED                         [ 78%]\n"
            "utils/test_strings.py::test_slugify PASSED                     [ 85%]\n"
            "utils/test_strings.py::test_camel_to_snake PASSED              [ 92%]\n"
            "utils/test_strings.py::test_snake_to_camel PASSED              [100%]\n\n"
            "============================== 14 passed in 0.18s "
            "=============================="
        ),
    },
    # ── non-exact: large file reads ───────────────────────
    {
        "tool": "read_file",
        "input": {"filepath": "utils/config.py"},
        "raw": (
            "# Configuration module\n"
            "import os\n"
            "from pathlib import Path\n\n"
            "BASE_DIR = Path(__file__).resolve().parent.parent\n"
            "DATA_DIR = BASE_DIR / 'data'\n"
            "LOG_DIR  = BASE_DIR / 'logs'\n"
            "TMP_DIR  = BASE_DIR / 'tmp'\n\n"
            "DEFAULT_TIMEOUT   = 30\n"
            "MAX_RETRIES       = 3\n"
            "BATCH_SIZE        = 64\n"
            "LEARNING_RATE     = 1e-4\n"
            "DROPOUT_RATE      = 0.1\n"
            "EMBEDDING_DIM     = 256\n"
            "HIDDEN_DIM        = 512\n"
            "NUM_LAYERS        = 6\n"
            "NUM_HEADS         = 8\n"
            "MAX_SEQ_LEN       = 2048\n"
            "VOCAB_SIZE        = 32000\n"
            "PAD_TOKEN_ID      = 0\n"
            "BOS_TOKEN_ID      = 1\n"
            "EOS_TOKEN_ID      = 2\n"
            "UNK_TOKEN_ID      = 3\n\n"
            "DB_HOST     = os.getenv('DB_HOST', 'localhost')\n"
            "DB_PORT     = int(os.getenv('DB_PORT', '5432'))\n"
            "DB_NAME     = os.getenv('DB_NAME', 'myapp')\n"
            "DB_USER     = os.getenv('DB_USER', 'admin')\n"
            "DB_PASSWORD = os.getenv('DB_PASSWORD', '')\n\n"
            "REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')\n"
            "REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))\n"
            "REDIS_DB   = int(os.getenv('REDIS_DB', '0'))\n\n"
            "LOG_LEVEL  = os.getenv('LOG_LEVEL', 'INFO')\n"
            "LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'\n\n"
            "ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')\n"
            "SECRET_KEY    = os.getenv('SECRET_KEY', 'dev-secret-do-not-use')\n"
            "DEBUG         = os.getenv('DEBUG', 'false').lower() == 'true'\n"
        ),
    },
    {
        "tool": "read_file",
        "input": {"filepath": "utils/validators.py"},
        "raw": (
            "import re\n"
            "from typing import Any\n\n"
            "EMAIL_RE    = re.compile(r'^[\\w.+-]+@[\\w-]+\\.[\\w.]+$')\n"
            "PHONE_RE    = re.compile(r'^\\+?[1-9]\\d{7,14}$')\n"
            "USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]{3,32}$')\n"
            "SLUG_RE     = re.compile(r'^[a-z0-9-]+$')\n\n"
            "def is_email(value: Any) -> bool:\n"
            "    return isinstance(value, str) and bool(EMAIL_RE.match(value))\n\n"
            "def is_phone(value: Any) -> bool:\n"
            "    return isinstance(value, str) and bool(PHONE_RE.match(value))\n\n"
            "def is_username(value: Any) -> bool:\n"
            "    return isinstance(value, str) and bool(USERNAME_RE.match(value))\n\n"
            "def is_slug(value: Any) -> bool:\n"
            "    return isinstance(value, str) and bool(SLUG_RE.match(value))\n\n"
            "def is_positive_int(value: Any) -> bool:\n"
            "    return isinstance(value, int) and value > 0\n\n"
            "def is_non_empty_string(value: Any) -> bool:\n"
            "    return isinstance(value, str) and len(value.strip()) > 0\n\n"
            "def validate_range(value: float, lo: float, hi: float) -> bool:\n"
            "    return lo <= value <= hi\n\n"
            "def validate_length(s: str, min_len: int, max_len: int) -> bool:\n"
            "    return min_len <= len(s) <= max_len\n\n"
            "def validate_choices(value: Any, choices: list) -> bool:\n"
            "    return value in choices\n"
        ),
    },
    # ── non-exact: write confirmations ────────────────────
    {
        "tool": "write_file",
        "input": {"filepath": "utils/config.py",
                  "content": "# updated"},
        "raw": "Written 1842 chars to utils/config.py",
    },
    {
        "tool": "write_file",
        "input": {"filepath": "utils/validators.py",
                  "content": "# updated"},
        "raw": "Written 963 chars to utils/validators.py",
    },
    # ── non-exact: verbose passing run after a fix ────────
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_validators.py",
                  "code_file": "utils/validators.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 11 items\n\n"
            "utils/test_validators.py::test_is_email_valid PASSED           [  9%]\n"
            "utils/test_validators.py::test_is_email_invalid PASSED         [ 18%]\n"
            "utils/test_validators.py::test_is_phone_valid PASSED           [ 27%]\n"
            "utils/test_validators.py::test_is_phone_short PASSED           [ 36%]\n"
            "utils/test_validators.py::test_is_username PASSED              [ 45%]\n"
            "utils/test_validators.py::test_is_slug PASSED                  [ 54%]\n"
            "utils/test_validators.py::test_positive_int PASSED             [ 63%]\n"
            "utils/test_validators.py::test_non_empty_string PASSED         [ 72%]\n"
            "utils/test_validators.py::test_validate_range PASSED           [ 81%]\n"
            "utils/test_validators.py::test_validate_length PASSED          [ 90%]\n"
            "utils/test_validators.py::test_validate_choices PASSED         [100%]\n\n"
            "============================== 11 passed in 0.09s "
            "=============================="
        ),
    },
    {
        "tool": "read_file",
        "input": {"filepath": "utils/logger.py"},
        "raw": (
            "import logging\n"
            "import sys\n"
            "from logging.handlers import RotatingFileHandler\n\n"
            "def get_logger(name: str, level: str = 'INFO') -> logging.Logger:\n"
            "    logger = logging.getLogger(name)\n"
            "    if logger.handlers:\n"
            "        return logger\n"
            "    logger.setLevel(getattr(logging, level.upper(), logging.INFO))\n"
            "    fmt = logging.Formatter(\n"
            "        '%(asctime)s %(levelname)-8s %(name)s %(message)s',\n"
            "        datefmt='%Y-%m-%d %H:%M:%S'\n"
            "    )\n"
            "    ch = logging.StreamHandler(sys.stdout)\n"
            "    ch.setFormatter(fmt)\n"
            "    logger.addHandler(ch)\n"
            "    fh = RotatingFileHandler(\n"
            "        'app.log', maxBytes=10*1024*1024, backupCount=5\n"
            "    )\n"
            "    fh.setFormatter(fmt)\n"
            "    logger.addHandler(fh)\n"
            "    return logger\n\n"
            "def log_call(logger: logging.Logger):\n"
            "    import functools\n"
            "    def decorator(fn):\n"
            "        @functools.wraps(fn)\n"
            "        def wrapper(*args, **kwargs):\n"
            "            logger.debug('calling %s', fn.__name__)\n"
            "            result = fn(*args, **kwargs)\n"
            "            logger.debug('%s returned %r', fn.__name__, result)\n"
            "            return result\n"
            "        return wrapper\n"
            "    return decorator\n"
        ),
    },
    # ── exact: failing tests with specific load-bearing info ─
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_parser.py",
                  "code_file": "utils/parser.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 3 items\n\n"
            "utils/test_parser.py::test_parse_date FAILED                   [ 33%]\n"
            "utils/test_parser.py::test_parse_amount FAILED                 [ 66%]\n"
            "utils/test_parser.py::test_parse_status PASSED                 [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "____________________________ test_parse_date _____________________________\n"
            "utils/test_parser.py:8: in test_parse_date\n"
            "    assert parse_date('2024-03-15') == datetime(2024, 15, 3)\n"
            "AssertionError: assert datetime(2024, 3, 15, 0, 0) == datetime(2024, 15, 3)\n"
            "E   AssertionError: assert datetime(2024, 3, 15) != datetime(2024, 15, 3)\n"
            "____________________________ test_parse_amount ___________________________\n"
            "utils/test_parser.py:14: in test_parse_amount\n"
            "    assert parse_amount('$1,234.56') == 123456\n"
            "AssertionError: assert 1234.56 == 123456\n"
            "E   AssertionError: assert 1234.56 != 123456\n"
            "=========================== short test summary info ========================\n"
            "FAILED utils/test_parser.py::test_parse_date\n"
            "FAILED utils/test_parser.py::test_parse_amount\n"
            "========================= 2 failed, 1 passed in 0.07s ======================"
        ),
    },
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_cache.py",
                  "code_file": "utils/cache.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 4 items\n\n"
            "utils/test_cache.py::test_set_get FAILED                       [ 25%]\n"
            "utils/test_cache.py::test_expire FAILED                        [ 50%]\n"
            "utils/test_cache.py::test_delete PASSED                        [ 75%]\n"
            "utils/test_cache.py::test_clear PASSED                         [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "______________________________ test_set_get ________________________________\n"
            "    cache.set('mykey', {'user_id': 42, 'role': 'editor'}, ttl=300)\n"
            "    assert cache.get('mykey')['role'] == 'admin'\n"
            "AssertionError: assert 'editor' == 'admin'\n"
            "E   AssertionError: assert 'editor' != 'admin'\n"
            "______________________________ test_expire _________________________________\n"
            "    cache.set('token', 'abc123', ttl=1)\n"
            "    time.sleep(2)\n"
            "KeyError: 'token'\n"
            "E   KeyError: 'token'\n"
            "=========================== short test summary info ========================\n"
            "FAILED utils/test_cache.py::test_set_get\n"
            "FAILED utils/test_cache.py::test_expire\n"
            "========================= 2 failed, 2 passed in 2.14s ======================"
        ),
    },
    {
        "tool": "run_tests",
        "input": {"test_file": "utils/test_serializer.py",
                  "code_file": "utils/serializer.py"},
        "raw": (
            "============================= test session starts "
            "==============================\n"
            "platform darwin -- Python 3.12.7, pytest-8.3.2\n"
            "collecting ... collected 2 items\n\n"
            "utils/test_serializer.py::test_serialize FAILED                [ 50%]\n"
            "utils/test_serializer.py::test_deserialize FAILED              [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "___________________________ test_serialize _________________________________\n"
            "    result = serialize({'ts': 1712000000, 'uid': 'u_9f3a', 'evt': 'login'})\n"
            "    assert result['timestamp'] == 1712000000\n"
            "KeyError: 'timestamp'\n"
            "E   KeyError: 'timestamp'\n"
            "___________________________ test_deserialize _______________________________\n"
            "    obj = deserialize({'timestamp': 1712000000, 'user_id': 'u_9f3a'})\n"
            "    assert obj.ts == 1712000000\n"
            "AttributeError: 'Event' object has no attribute 'ts'\n"
            "E   AttributeError: 'Event' object has no attribute 'ts'\n"
            "=========================== short test summary info ========================\n"
            "FAILED utils/test_serializer.py::test_serialize\n"
            "FAILED utils/test_serializer.py::test_deserialize\n"
            "========================= 2 failed in 0.05s =============================="
        ),
    },
]

EXACTNESS_PATTERNS = [
    r'Traceback \(most recent call last\)',
    r'File ".+", line \d+',
    r'AssertionError',
    r'KeyError:',
    r'TypeError:',
    r'ValueError:',
    r'AttributeError:',
    r'RuntimeError:',
    r'ModuleNotFoundError:',
    r'unexpected keyword argument',
    r'No module named',
    r"'[^']+' != '[^']+'",
    r'expected.+got',
    r'FAILED|ERROR',
]

def classify_exactness(tool_name: str,
                        content: str) -> tuple[bool, float]:
    base = 0.2
    if tool_name in LIKELY_EXACT_TOOLS:
        threshold = base - 0.10
    elif tool_name in LIKELY_NONEXACT_TOOLS:
        threshold = base + 0.1
    else:
        threshold = base

    hits = sum(
        1 for p in EXACTNESS_PATTERNS
        if re.search(p, content, re.MULTILINE)
    )
    confidence = round(hits / len(EXACTNESS_PATTERNS), 3)
    return confidence >= threshold, confidence

# ── Summarizer ────────────────────────────────────────────

def summarize(content: str,
              extra_token_tracker: list) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this tool output in 2-3 sentences. "
                "Preserve key findings. Remove verbose details.\n\n"
                f"Output:\n{content[:3000]}\n\nSummary:"
            )
        }]
    )
    extra_token_tracker[0] += response.usage.input_tokens
    extra_token_tracker[0] += response.usage.output_tokens
    return response.content[0].text.strip()

# ── Triage Gate ───────────────────────────────────────────

def triage(tool_name: str, content: str,
           condition: str,
           extra_token_tracker: list) -> tuple[str, bool, float]:
    exactness, confidence = classify_exactness(tool_name, content)

    if condition == "full_exact":
        return content, exactness, confidence
    elif condition == "summarize_all":
        return (summarize(content, extra_token_tracker),
                exactness, confidence)
    elif condition == "exactness_aware":
        if exactness:
            return content, exactness, confidence
        return (summarize(content, extra_token_tracker),
                exactness, confidence)

    raise ValueError(f"Unknown condition: {condition}")

# ── Tools ─────────────────────────────────────────────────

def run_tests_with_status(test_file: str,
                           code_file: str) -> tuple[str, bool]:
    task_dir = os.path.dirname(os.path.abspath(test_file))
    test_filename = os.path.basename(test_file)
    result = subprocess.run(
        ["python", "-m", "pytest", test_filename, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30,
        cwd=task_dir
    )
    output = (result.stdout + result.stderr)[:4000]
    passed = result.returncode == 0
    return output, passed

def read_file(filepath: str) -> str:
    if re.search(r'test_.*\.py$',
                 os.path.basename(filepath)):
        return ("Error: test files cannot be read. "
                "Use run_tests to execute tests and see output.")
    try:
        with open(filepath) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(filepath: str, content: str) -> str:
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Written {len(content)} chars to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"

TOOL_DEFINITIONS = [
    {
        "name": "run_tests",
        "description": (
            "Run the pytest test suite. Returns test output "
            "including pass/fail status and error messages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_file": {"type": "string"},
                "code_file": {"type": "string"}
            },
            "required": ["test_file", "code_file"]
        }
    },
    {
        "name": "read_file",
        "description": (
            "Read a source code file. "
            "Cannot be used on test files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["filepath", "content"]
        }
    }
]

# ── Agent Loop ────────────────────────────────────────────

def build_pressure_history(condition: str,
                           extra_token_tracker: list,
                           conn, session_id: str,
                           task_id: str, task_type: str,
                           replicate: int) -> tuple[list, int]:
    """
    Build synthetic past tool-call message pairs to pre-fill context.
    Each noise observation goes through the triage gate so the
    condition difference in context volume is preserved.
    Returns (messages, ctx_tokens_added).
    """
    messages = []
    ctx_tokens = 0
    for i, obs in enumerate(PRESSURE_NOISE):
        fake_id = str(uuid.uuid4())
        tool_name = obs["tool"]
        raw = obs["raw"]

        processed, exactness, confidence = triage(
            tool_name, raw, condition, extra_token_tracker
        )
        proc_tok = log_observation(
            conn, session_id, task_id, task_type,
            replicate, -(i + 1), tool_name,
            raw, processed, exactness, confidence, condition
        )
        ctx_tokens += proc_tok

        messages.append({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": fake_id,
                "name": tool_name,
                "input": obs["input"],
            }]
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": fake_id,
                "content": processed,
            }]
        })

    return messages, ctx_tokens


SYSTEM_PROMPT = """You are a coding agent that fixes bugs in Python files.

Tools available:
- read_file: read source code files (not test files)
- write_file: write a corrected version of a file
- run_tests: run the test suite and see failure output

Workflow:
1. Read the buggy file to understand the code
2. Run the tests to see the exact failure output
3. Analyze the error carefully
4. Write the corrected file
5. Run tests again to confirm all pass

Fix only what is necessary. Do not modify test files."""

def run_agent(task: dict, condition: str,
              replicate: int,
              conn: sqlite3.Connection,
              pressure: bool = False) -> dict:
    session_id = str(uuid.uuid4())
    task_id = task["id"]
    task_type = task["type"]
    start_time = time.time()

    main_billed_tokens = 0
    extra_token_tracker = [0]
    ctx_token_volume = 0

    initial_msg = {
        "role": "user",
        "content": (
            f"Fix the bug in {task['code_file']}.\n"
            f"Tests are in {task['test_file']}.\n"
            f"Make all tests pass. "
            f"Do not modify the test file."
        )
    }

    if pressure:
        noise_msgs, noise_tokens = build_pressure_history(
            condition, extra_token_tracker, conn,
            session_id, task_id, task_type, replicate
        )
        ctx_token_volume += noise_tokens
        messages = [initial_msg] + noise_msgs
    else:
        messages = [initial_msg]
    step = 0
    max_steps = 10
    passed = False

    while step < max_steps:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            temperature=0,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages
        )

        main_billed_tokens += response.usage.input_tokens
        main_billed_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            _, passed = run_tests_with_status(
                task["test_file"], task["code_file"]
            )
            break

        messages.append({
            "role": "assistant",
            "content": response.content
        })

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "run_tests":
                raw, _ = run_tests_with_status(
                    block.input["test_file"],
                    block.input["code_file"]
                )
            elif block.name == "read_file":
                raw = read_file(block.input["filepath"])
            elif block.name == "write_file":
                raw = write_file(
                    block.input["filepath"],
                    block.input["content"]
                )
            else:
                raw = "Unknown tool"

            processed, exactness, confidence = triage(
                block.name, raw, condition, extra_token_tracker
            )

            proc_tok = log_observation(
                conn, session_id, task_id, task_type,
                replicate, step, block.name,
                raw, processed, exactness,
                confidence, condition
            )
            ctx_token_volume += proc_tok

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": processed
            })

        messages.append({
            "role": "user",
            "content": tool_results
        })
        step += 1

    latency = round(time.time() - start_time, 2)
    total_billed = main_billed_tokens + extra_token_tracker[0]

    log_session(
        conn, session_id, task_id, task_type,
        replicate, condition, passed,
        ctx_token_volume, total_billed, latency, step
    )

    return {
        "session_id": session_id,
        "task_id": task_id,
        "task_type": task_type,
        "replicate": replicate,
        "condition": condition,
        "pressure": pressure,
        "passed": passed,
        "approx_ctx_token_volume": ctx_token_volume,
        "total_billed_tokens": total_billed,
        "latency_seconds": latency,
        "steps": step
    }
