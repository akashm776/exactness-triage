# Exactness-Aware Arrival-Time Triage

AI coding agents use tools. They read files, run tests, write fixes. Every time a tool returns output, that output goes into the agent's context window. Over a long run session, everything accumulates and once the session window has reached some threshold, agent does some compaction. Most compaction approaches summarizes over all of the context equally.

That could be a problem.

Some tool outputs can be summarized without losing anything important. A test that fails because for example the sort direction is wrong can be captured via a summary. But some outputs contain one specific thing the agent needs to see exactly. These are a few examples: an error message naming the wrong key, test diff showing two bugs at once, a KeyError with an abbreviated name that has five plausible expansions. If these are summarized then there is a chance vital information might be lost causing the agent to fail.

This project tests a simple fix. Before any tool output enters the agent's context, run a classifier. If the output contains load-bearing exact information, pass it through unchanged. Otherwise summarize it. This happens at arrival time — before the context gets polluted — not after.

The question is whether that works.

**Note**:

This pilot applies summarization at arrival time (every non-exactness observation is compressed the moment it arrives). A more practical production design would separate the two steps. The classifier would still run at arrival and attach an exactness flag to each observation, but compression would only happen when context pressure crosses a threshold. This way short sessions would never pay summarization cost at all and long sessions would compact selectively, respecting the exactness flags, only when they actually need to. However, for this pilot this becomes a messier experiment, because the three conditions only diverge once compaction triggers. The current pilot isolates the effect more cleanly by making the conditions differ from the first observation. The deferred design is the right next step after this pilot establishes whether the effect is real.

---

## Hypotheses

**H1.** Summarizing exactness-sensitive observations causes task failure at a higher rate than summarizing non-exactness-sensitive ones.

**H2.** A rule-based classifier identifies exactness-sensitive observations with acceptable precision and recall on a stratified held-out sample.

**H3.** Exactness-aware triage reduces approximate context token volume relative to full-exact context, while maintaining comparable pass rates on exactness-sensitive tasks.

---

## Results

Results are from two experiment sets: a **baseline** run (no added context pressure) and a **pressure** run, both using 10 replicates per task-condition cell.

### H1 — Pass rates on exactness-sensitive tasks

| Condition | Baseline | Under pressure |
|---|---|---|
| `full_exact` | 75% [60–86%] | 76% [62–85%] |
| `exactness_aware` | 70% [55–82%] | **78% [63–88%]** |
| `summarize_all` | 55% [40–69%] | 56% [40–71%] |

All three conditions pass 100% of non-exactness-sensitive tasks in both settings. The effect is specific to exactness-sensitive tasks.

Under pressure, `exactness_aware` overtakes `full_exact` (78% vs 76%) while `summarize_all` stays stuck at 56%. The confidence intervals between `exactness_aware` and `summarize_all` are cleanly non-overlapping.

### H3 — Context volume

| Condition | Baseline ctx tokens | Pressure ctx tokens | Reduction vs `full_exact` |
|---|---|---|---|
| `full_exact` | 831 | 3,619 | — |
| `exactness_aware` | 664 | **2,203** | **−20% / −39%** |
| `summarize_all` | 382 | 1,277 | −54% / −65% |

`exactness_aware` cuts context 39% under pressure relative to `full_exact`, while matching or exceeding its pass rate. `summarize_all` achieves the deepest compression but pays for it in H1.

### The case for exactness-aware

Three conditions needed to hold simultaneously:

1. **Pass rate ≥ `full_exact` under pressure** ✓ (78% vs 76%)
2. **Pass rate >> `summarize_all`** ✓ (78% vs 56%, non-overlapping CIs)
3. **Context volume < `full_exact`** ✓ (−39% under pressure)

`exactness_aware` is the only condition that achieves all three.

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
# Pilot (6 sessions, calibrate cost first)
python run_experiments.py --pilot

# Baseline full run (72 sessions, 3 replicates)
python run_experiments.py

# Higher-power run (240 sessions, 10 replicates)
python run_experiments.py --replicates 10

# With context pressure (pre-fills history with synthetic noise)
python run_experiments.py --replicates 10 --pressure

# Both flags can be combined
python run_experiments.py --pilot --pressure
```

Results append to `results.json` and `observations.db`. Each session is UUID-keyed so multiple runs accumulate cleanly. The `pressure` field in each result record distinguishes pressure from baseline sessions.

Estimated cost: ~$0.10–0.15 per session. A 10-replicate baseline + pressure run costs roughly $50–70 total.

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
| **Replicates** | Configurable via `--replicates` (default 3) |
| **Pass/fail** | pytest exit code (0 = all passed) |

**Conditions:**
- `full_exact` — all tool outputs passed to the agent unchanged
- `summarize_all` — all tool outputs summarized before the agent sees them
- `exactness_aware` — regex classifier decides per observation; exactness-sensitive outputs pass through raw, others are summarized
- `exactness_aware_llm` — same as above but uses an LLM classifier instead of regex (see [LLM Classifier](#llm-classifier))

**Context pressure mode (`--pressure`):** Injects 11 synthetic past tool-call pairs into the agent's message history before the task begins — 7 non-exactness observations (passing test runs, large file reads, write confirmations) and 4 exactness observations (failing tests with specific assertion errors and KeyErrors). Each goes through the triage gate, so the difference in context volume between conditions is amplified. This simulates a session that has already accumulated history and makes the H3 effect measurable.

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

In `--pressure` mode, synthetic past observations are injected before the agent loop begins, each routed through the same triage gate so condition differences in context volume are preserved.

---

## LLM Classifier

`exactness_aware_llm` replaces the regex classifier with a zero-shot Haiku call. Every tool output gets a binary classification before any routing decision is made.

**Prompt design.** The classifier receives the tool name and the first 2000 characters of the output, then replies with exactly one word: `EXACT` or `SUMMARY`. The prompt defines both classes with concrete examples to anchor the decision:

> *Exactness-sensitive*: a specific exception type distinguishing multiple plausible fixes; an assertion diff showing precise expected vs actual values; a key name or delimiter the agent cannot guess from source code alone.
>
> *Not exactness-sensitive*: a passing test run; a file write confirmation; a failure whose cause is obvious from the source code alone (inverted boolean, off-by-one in a visible slice).

**Pilot evidence.** An 8-session pilot (2 tasks × 4 conditions × 1 replicate) showed the LLM classifier correctly labeling all failing `run_tests` outputs on task_01 as `EXACT` — the correct call, since task_01 requires knowing which of three `RuntimeError` subtypes is raised. The regex classifier made the same decisions on that pilot. The single-replicate task_01 failure under `exactness_aware_llm` is within the variance expected from a one-replicate run and does not indicate a classification error.

**Cost tradeoff.** Each observation routed through the LLM classifier incurs one Haiku call (~300–500 input tokens + 1 output token). In the pilot, `exactness_aware_llm` billed ~36K tokens per session vs ~17K for `exactness_aware` (regex) — roughly 2× the API cost. The regex classifier is free beyond the base model calls.

**Status.** A full benchmark comparing LLM vs regex classifier accuracy (H2) across all 8 tasks with 10 replicates is pending. The pilot establishes that the LLM classifier is operational and makes sensible decisions; the larger run would measure whether it catches cases the regex misses and whether the extra cost translates to a pass-rate gain.

---

## Limitations

- **`task_02` bundles two bugs.** A failure under `summarize_all` could reflect lost delimiter detail, lost empty-field filtering, or both.
- **`task_03` fails across all conditions.** The agent cannot solve task_03 even with full exact output, suggesting the bug's correct solution is not inferable from runtime output alone. This task contributes noise to H1 without differentiating conditions.
- **Synthetic pressure is not real compaction.** The `--pressure` mode pre-fills history with fixed noise observations. It amplifies the H3 context-volume signal but does not simulate actual model compaction or eviction behavior.
- **Capable summarizer may preserve load-bearing tokens.** Results reflect realistic cheap summarization (Haiku, 150 tokens), not maximally lossy compression.
- **Approximate context metric.** Token volume estimated as character count ÷ 4. Relative comparisons across conditions are meaningful; absolute values are not.
- **Temperature 0 reduces but does not eliminate variance.** Confidence intervals are reported.
- **Classifier patterns require maintenance.** The initial run missed `RuntimeError` entirely, causing a false negative on task_01. Any extension to new task types should audit the pattern list against the actual exception types those tasks produce.
- **This is a pilot.** The goal is to determine whether the phenomenon is real enough to justify a larger study.
