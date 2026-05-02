# Exactness-Aware Arrival-Time Triage

AI coding agents use tools. They read files, run tests, inspect errors, and write fixes. Every tool output becomes part of the agent's working context. In long sessions, this creates pressure: either the context grows too large, or the agent/runtime has to compact previous observations.

Most compaction treats observations uniformly. This project tests a narrower idea:

> Some tool outputs can be safely summarized. Others contain exact, load-bearing details that should be preserved verbatim.

Examples of exactness-sensitive information include:

- a `KeyError` naming the missing key
- an assertion diff showing the expected and actual values
- an exception type that distinguishes between multiple plausible fixes
- a test failure where one abbreviation, delimiter, or field name determines the solution

Uniform summarization can erase or blur these details. This repo tests **arrival-time exactness triage**: classify each tool output before it enters the agent's context. If the observation appears exactness-sensitive, preserve it raw. Otherwise, summarize it.

The question is simple:

> Can exactness-aware triage reduce context volume without sacrificing task success?

---

## Core Idea

A normal coding-agent loop looks like this:

```text
Agent calls tool
      ↓
Tool returns output
      ↓
Output enters context
      ↓
Agent continues
```

This project inserts one gate:

```text
Agent calls tool
      ↓
Tool returns raw output
      ↓
[Exactness Triage Gate]
      ├── exactness-sensitive → keep raw
      └── not exactness-sensitive → summarize
      ↓
Processed output enters context
```

The gate runs at **arrival time**, before the context is polluted with unnecessary detail.

A more production-like design would likely defer summarization until context pressure is actually high. In that setting, the arrival-time classifier would attach metadata to each observation, and later compaction would use those flags. This pilot compresses immediately so the three experimental conditions differ from the first observation, making the effect easier to measure.

---

## Hypotheses

**H1.** Uniformly summarizing exactness-sensitive observations hurts task success more than summarizing non-exactness-sensitive observations.

**H2.** A simple rule-based classifier can identify exactness-sensitive tool outputs with reasonable precision and recall.

**H3.** Exactness-aware triage reduces approximate context volume relative to full raw context while maintaining better task success than uniform summarization.

---

## Results Summary

The current pilot includes two experiment settings:

1. **Baseline:** normal agent runs with no extra context pressure.
2. **Pressure:** synthetic prior tool observations are injected into the conversation before the task begins.

Each setting uses:

```text
8 tasks × 3 conditions × 10 replicates = 240 sessions
```

The task set contains 4 exactness-sensitive tasks and 4 non-exactness-sensitive tasks.

### H1: Exactness-Sensitive Pass Rate

| Condition         |     Baseline | Under Pressure |
| ----------------- | -----------: | -------------: |
| `full_exact`      | 75% [60–86%] |   76% [62–85%] |
| `exactness_aware` | 70% [55–82%] |   78% [63–88%] |
| `summarize_all`   | 55% [40–69%] |   56% [40–71%] |

All three conditions achieved 100% pass rate on non-exactness-sensitive tasks in both settings.

The important pattern is that `summarize_all` is consistently worse on exactness-sensitive tasks, while `exactness_aware` remains close to `full_exact`. Under pressure, `exactness_aware` slightly outperforms `full_exact` in this run, but the confidence intervals overlap, so this should be interpreted as **comparable pass rate**, not a definitive win.

The stronger claim is:

> `exactness_aware` preserved task success much better than `summarize_all` on exactness-sensitive tasks.

### H3: Context Volume

Approximate context volume is estimated using character count divided by 4. This is not an exact tokenizer measurement, but it is useful for relative comparisons across conditions.

| Condition         | Baseline Context Tokens | Pressure Context Tokens | Reduction vs `full_exact` |
| ----------------- | ----------------------: | ----------------------: | ------------------------: |
| `full_exact`      |                     831 |                   3,619 |                         — |
| `exactness_aware` |                     664 |                   2,203 |               −20% / −39% |
| `summarize_all`   |                     382 |                   1,277 |               −54% / −65% |

`summarize_all` compresses the most, but loses task success on exactness-sensitive problems. `full_exact` preserves all details, but carries the largest context. `exactness_aware` gives the best tradeoff in this pilot: substantially lower context than `full_exact`, with much better pass rate than `summarize_all`.

### Current Interpretation

The pilot supports the core idea:

> Exactness-aware triage is a better default than uniform summarization when tool outputs may contain load-bearing exact details.

The result should still be treated as a pilot finding, not a broad benchmark result. The task set is small, synthetic, and designed to isolate the failure mode.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Running Experiments

### Pilot Run

Use this first to verify setup and estimate cost.

```bash
python run_experiments.py --pilot
```

This runs:

```text
2 tasks × 3 conditions × 1 replicate = 6 sessions
```

### Baseline Full Run

```bash
python run_experiments.py --replicates 10
```

### Pressure Run

```bash
python run_experiments.py --replicates 10 --pressure
```

### Important: Keep Runs Separate

The experiment script appends to `results.json`. For clean analysis, run baseline and pressure experiments separately and save their outputs.

Recommended workflow:

```bash
# Baseline
rm -f results.json observations.db
python run_experiments.py --replicates 10
mkdir -p results
cp results.json results/baseline_results.json
cp observations.db results/baseline_observations.db

# Pressure
rm -f results.json observations.db
python run_experiments.py --replicates 10 --pressure
cp results.json results/pressure_results.json
cp observations.db results/pressure_observations.db
```

---

## Analysis

```bash
python analyze.py
```

The analysis script reports:

* pass rates by condition and task type
* Wilson confidence intervals
* approximate context volume
* billed token averages
* latency averages
* failure examples from `summarize_all`
* annotation samples for classifier evaluation

To evaluate H2, run `analyze.py` first to generate `annotation_sample.csv`, then manually label each row by adding a `human_exactness` column:

```text
1 = exactness-sensitive
0 = not exactness-sensitive
```

Then rerun `analyze.py`. The script will report classifier precision, recall, and false negatives.

---

## Experimental Design

| Field                   | Value                                            |
| ----------------------- | ------------------------------------------------ |
| Agent model             | Claude Sonnet 4.6                                |
| Summarizer model        | Claude Haiku                                     |
| Agent temperature       | 0                                                |
| Max agent steps         | 10                                               |
| Tasks                   | 8 single-file Python bug-fix tasks               |
| Task split              | 4 exactness-sensitive, 4 non-exactness-sensitive |
| Conditions              | `full_exact`, `summarize_all`, `exactness_aware` |
| Default replicates      | 3                                                |
| Higher-power replicates | 10                                               |
| Pass/fail metric        | pytest exit code                                 |

### Conditions

| Condition         | Behavior                                                     |
| ----------------- | ------------------------------------------------------------ |
| `full_exact`      | Pass every tool output to the agent unchanged                |
| `summarize_all`   | Summarize every tool output before the agent sees it         |
| `exactness_aware` | Preserve exactness-sensitive outputs raw; summarize the rest |

### Key Constraint

The agent cannot read test files directly. It can only observe test expectations through runtime output from `run_tests`. This makes exactness-sensitive tasks genuinely dependent on tool-output details.

---

## Task Set

### Exactness-Sensitive Tasks

| Task      | Bug                                               | Why Exactness Matters                                                                     |
| --------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `task_01` | Wrong exception type among multiple candidates    | A generic summary may say "wrong exception type" without preserving which one is expected |
| `task_02` | Wrong delimiter and missing empty-field filtering | Two separate bugs are revealed through exact test output                                  |
| `task_03` | Abbreviated key names                             | Keys such as `ts`, `uid`, `evt`, and `lvl` have multiple plausible expansions             |
| `task_04` | Field swap and missing zero-padding               | The correct field direction and padding requirements are not obvious from code alone      |

### Non-Exactness-Sensitive Tasks

| Task      | Bug                         | Why Summary Suffices                                            |
| --------- | --------------------------- | --------------------------------------------------------------- |
| `task_05` | Inverted boolean            | "Condition is inverted" plus visible source code is enough      |
| `task_06` | Off-by-one slice            | "Slice excludes last middle element" plus source code is enough |
| `task_07` | Wrong accumulator operation | "Uses addition instead of multiplication" is enough             |
| `task_08` | Missing base case           | "Infinite recursion / missing base case" is enough              |

---

## Context Pressure Mode

The `--pressure` flag injects synthetic prior tool observations before the main task begins.

These include:

* verbose passing test runs
* large but mostly irrelevant file reads
* write confirmations
* failing test outputs with exact assertion and exception details

Each synthetic observation passes through the same triage gate as real tool outputs. This amplifies context-volume differences between the three conditions.

This is not meant to fully simulate production context compaction. It is a controlled stressor that makes the cost/correctness tradeoff easier to observe.

---

## Repository Structure

```text
exactness-triage/
├── tasks/
│   ├── task_01/
│   │   ├── buggy.py
│   │   ├── buggy_original.py
│   │   ├── test_buggy.py
│   │   └── meta.json
│   ├── task_02/
│   └── ...
├── agent.py
├── run_experiments.py
├── analyze.py
├── requirements.txt
├── results.json              # generated
├── observations.db           # generated
└── annotation_sample.csv     # generated
```

---

## Implementation Overview

### `agent.py`

Contains:

* the coding-agent loop
* tool definitions
* test execution
* file read/write helpers
* exactness classifier
* triage logic
* SQLite logging
* pressure-history construction

### `run_experiments.py`

Runs the outer experiment loop across:

* tasks
* conditions
* replicates
* baseline vs pressure mode

### `analyze.py`

Computes:

* pass-rate tables
* Wilson confidence intervals
* context-volume averages
* latency and billed-token summaries
* failure examples
* annotation samples for H2

---

## Classifier

The current classifier is rule-based. It uses tool priors and regex patterns for outputs likely to contain exact details, such as:

* `AssertionError`
* `KeyError`
* `TypeError`
* `ValueError`
* `AttributeError`
* `RuntimeError`
* `ModuleNotFoundError`
* pytest `FAILED` / `ERROR`
* expected-vs-actual diffs

This is intentionally simple. The point of the pilot is not to prove that this classifier is optimal. The point is to test whether exactness-sensitive observations are worth treating differently at all.

---

## Limitations

* **Small benchmark.** The current task set has only 8 tasks.
* **Synthetic tasks.** The benchmark is controlled and useful for isolating the phenomenon, but it does not prove generality across real repositories.
* **Synthetic pressure.** Pressure mode injects fixed prior observations. It does not simulate real long-horizon compaction, eviction, or memory decay.
* **Confidence intervals overlap.** The pilot supports a strong directional result, but some condition comparisons are not statistically decisive.
* **`task_02` bundles two bugs.** A failure may reflect loss of delimiter detail, loss of empty-field filtering, or both.
* **`task_03` is noisy.** The agent fails even with full exact output, suggesting the task may be too underdetermined.
* **Approximate token volume.** Context volume is estimated using character count divided by 4, not a model-specific tokenizer.
* **Classifier needs maintenance.** New task families may require additional patterns or a learned classifier.
* **Summarizer quality matters.** A stronger or more verbose summarizer may preserve more exact details, reducing the observed gap.
* **This is a pilot.** The goal is to determine whether the phenomenon is real enough to justify a larger benchmark.

---

## Main Takeaway

Uniform summarization is too blunt for coding agents.

Some observations are safe to compress. Others contain exact details that determine whether the agent can solve the task. This pilot shows that a simple arrival-time exactness gate can preserve those details, reduce context volume, and avoid the failure mode introduced by summarizing everything.

The strongest current claim is:

> In a controlled coding-agent benchmark, exactness-aware triage preserved pass rate on exactness-sensitive tasks substantially better than uniform summarization while reducing context volume relative to full raw context.

---

## Next Steps

* Commit clean baseline and pressure result artifacts.
* Complete H2 manual annotation and report classifier precision/recall.
* Add real-world bug-fix tasks from small open-source repositories.
* Replace the rule-based classifier with a learned or LLM-based exactness classifier.
* Test deferred compaction: classify at arrival time, but summarize only when context pressure crosses a threshold.
* Evaluate on longer multi-step tasks where tool-output history matters more.
