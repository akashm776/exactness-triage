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

EXACTNESS_PATTERNS = [
    r'Traceback \(most recent call last\)',
    r'File ".+", line \d+',
    r'AssertionError',
    r'KeyError:',
    r'TypeError:',
    r'ValueError:',
    r'AttributeError:',
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
        threshold = base - 0.05
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
              conn: sqlite3.Connection) -> dict:
    session_id = str(uuid.uuid4())
    task_id = task["id"]
    task_type = task["type"]
    start_time = time.time()

    messages = [{
        "role": "user",
        "content": (
            f"Fix the bug in {task['code_file']}.\n"
            f"Tests are in {task['test_file']}.\n"
            f"Make all tests pass. "
            f"Do not modify the test file."
        )
    }]

    main_billed_tokens = 0
    extra_token_tracker = [0]
    ctx_token_volume = 0
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
        "passed": passed,
        "approx_ctx_token_volume": ctx_token_volume,
        "total_billed_tokens": total_billed,
        "latency_seconds": latency,
        "steps": step
    }
