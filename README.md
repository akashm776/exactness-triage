# Exactness-Aware Arrival-Time Triage

AI coding agents use tools. They read files, run tests, write fixes. Every time a tool returns output, that output goes into the agent's context window. Over a long run session, everything accumulates and once the session window has reached some threshold, agent does some compaction. Most compaction approaches summarizes over all of the context equally.

That could be a problem.

Some tool outputs can be summarized without losing anything important. A test that fails because for example the sort direction is wrong can be captured via a summary. But some outputs contain one specific thing the agent needs to see exactly. These are a few examples: an error message naming the wrong key, test diff showing two bugs at once, a KeyError with an abbreviated name that has five plausible expansions. If these are summarized then there is a chance vital information might be lost causing the agent to fail.

This project tests a simple fix. Before any tool output enters the agent's context, run a classifier. If the output contains load-bearing exact information, pass it through unchanged. Otherwise summarize it. This happens at arrival time — before the context gets polluted — not after.

The question is whether that works.

---

## Hypotheses

**H1.** Summarizing exactness-sensitive observations causes task failure at a higher rate than summarizing non-exactness-sensitive ones.

**H2.** A rule-based classifier identifies exactness-sensitive observations with acceptable precision and recall on a stratified held-out sample.

**H3.** Exactness-aware triage reduces approximate context token volume relative to full-exact context, while maintaining comparable pass rates on exactness-sensitive tasks.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Running the Experiment

```bash
python run_experiments.py
```

Runs 72 sessions: 8 tasks × 3 conditions × 3 replicates. Results are written to `results.json` and `observations.db`.

Before committing to the full run, test with 5 sessions first to calibrate cost (estimated $3–8 total at current pricing).

---

## Analysis

```bash
python analyze.py
```

Prints pass-rate tables with Wilson confidence intervals (H1), mean context token volume and billed tokens by condition (H3), failure examples from `summarize_all`, and exports `annotation_sample.csv` for manual H2 labeling.

To evaluate the classifier (H2), add a `human_exactness` column (0 or 1) to `annotation_sample.csv`, then re-run `analyze.py`.

---

## Experimental Design

| | |
|---|---|
| **Agent model** | Claude Sonnet 4.6, temperature 0, max 10 steps |
| **Summarizer model** | Claude Haiku, temperature 0, max 150 tokens |
| **Tasks** | 8 single-file Python bug-fix tasks |
| **Task split** | 4 exactness-sensitive, 4 non-exactness-sensitive |
| **Conditions** | `full_exact`, `summarize_all`, `exactness_aware` |
| **Replicates** | 3 per task-condition cell |
| **Total sessions** | 72 |
| **Pass/fail** | pytest exit code (0 = all passed) |

**Conditions:**
- `full_exact` — all tool outputs passed to the agent unchanged
- `summarize_all` — all tool outputs summarized before the agent sees them
- `exactness_aware` — classifier decides per observation at arrival time; exactness-sensitive observations pass through raw, others are summarized

**Key constraint:** The agent cannot read test files. Runtime output is the only source of test expectations, making exactness-sensitive tasks genuinely dependent on exact output.

---

## Task Set

### Exactness-Sensitive

| Task | Bug | Why exactness matters |
|---|---|---|
| `task_01` | Wrong exception type, 3 candidates | Summary "wrong exception type" leaves open which case and which target |
| `task_02` | Wrong delimiter + missing empty-field filter | Multiple plausible delimiters; two independent bugs both revealed by exact output |
| `task_03` | Abbreviated wrong key names | `ts`, `uid`, `evt`, `lvl` each have multiple plausible expansions |
| `task_04` | Field swap + missing zero-padding | Neither the swap direction nor which fields need padding are inferrable from code alone |

### Non-Exactness-Sensitive

| Task | Bug | Why summary suffices |
|---|---|---|
| `task_05` | Inverted boolean | "Condition is inverted" + visible `not (...)` → remove `not` |
| `task_06` | Off-by-one in slice | "Slice excludes last middle element" + visible `items[1:n-2]` is sufficient |
| `task_07` | Wrong accumulator operation | "Accumulates with addition instead of multiplication" + visible `result += n` is sufficient |
| `task_08` | Missing base case | "No base case, infinite recursion" + visible missing `if` statement is sufficient |

---

## Directory Structure

```
exactness-triage/
├── tasks/
│   ├── task_01/
│   │   ├── buggy.py             ← reset before each session
│   │   ├── buggy_original.py    ← never modified
│   │   ├── test_buggy.py
│   │   └── meta.json
│   ├── task_02/ ... task_08/
├── agent.py                     ← triage gate, classifier, agent loop, DB logging
├── run_experiments.py           ← outer experiment loop
├── analyze.py                   ← H1/H2/H3 analysis and annotation export
├── requirements.txt
├── observations.db              ← generated: per-observation archive
├── results.json                 ← generated: per-session summary
└── annotation_sample.csv        ← generated: stratified sample for H2 labeling
```

---

## Architecture

```
Agent calls tool
      ↓
Tool executes, returns raw output
      ↓
[Triage Gate]
  classify_exactness(tool_name, content)
  log raw + metadata to SQLite
  apply condition treatment
      ↓
Processed output enters agent message history
```

The triage gate is the only addition to a standard agent loop.

---

## Limitations

- **`task_02` bundles two bugs.** A failure under `summarize_all` could reflect lost delimiter detail, lost empty-field filtering, or both.
- **Small synthetic benchmark.** 8 tasks, 3 replicates. Results are directional; effect sizes may not generalize.
- **Capable summarizer may preserve load-bearing tokens.** Results reflect realistic cheap summarization, not maximally lossy compression.
- **Approximate context metric.** Token volume estimated as character count ÷ 4. Relative comparisons across conditions are meaningful; absolute values are not.
- **Temperature 0 reduces but does not eliminate variance.** Confidence intervals are reported.
- **This is a pilot.** The goal is to determine whether the phenomenon is real enough to justify a larger study.
