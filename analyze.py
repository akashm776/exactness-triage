import json
import sqlite3
import pandas as pd
import numpy as np

def load_results(path="results.json") -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))

def wilson_ci(s, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = s / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return round(center - margin, 3), round(center + margin, 3)

def pass_rate_table(df):
    rows = []
    for (cond, ttype), g in df.groupby(
            ["condition", "task_type"]):
        n = len(g)
        s = int(g["passed"].sum())
        lo, hi = wilson_ci(s, n)
        rows.append({
            "condition": cond,
            "task_type": ttype,
            "passed": s,
            "n": n,
            "pass_rate": round(s/n, 3),
            "95_ci": f"[{lo},{hi}]"
        })
    return pd.DataFrame(rows).set_index(
        ["condition", "task_type"]
    )

def cost_table(df):
    return df.groupby("condition").agg(
        approx_ctx_token_vol=("approx_ctx_token_volume", "mean"),
        total_billed_tokens=("total_billed_tokens", "mean"),
        latency_s=("latency_seconds", "mean")
    ).round(1)

def export_annotation_sample(db_path="observations.db",
                               n=80):
    """Stratified by tool_name and task_type."""
    conn = sqlite3.connect(db_path)
    strata = pd.read_sql("""
        SELECT tool_name, task_type, COUNT(*) as cnt
        FROM observations
        WHERE condition = 'full_exact'
        GROUP BY tool_name, task_type
    """, conn)

    total = strata["cnt"].sum()
    frames = []
    for _, row in strata.iterrows():
        k = max(1, round(n * row["cnt"] / total))
        frame = pd.read_sql(f"""
            SELECT tool_name, task_type,
                   substr(raw_content, 1, 400) as raw_preview,
                   exactness_label,
                   exactness_confidence
            FROM observations
            WHERE condition = 'full_exact'
              AND tool_name = '{row["tool_name"]}'
              AND task_type = '{row["task_type"]}'
            ORDER BY RANDOM()
            LIMIT {k}
        """, conn)
        frames.append(frame)

    conn.close()
    sample = pd.concat(frames).sample(frac=1).reset_index(
        drop=True
    )
    sample.to_csv("annotation_sample.csv", index=False)
    print(f"Exported {len(sample)} observations "
          f"(stratified) to annotation_sample.csv")
    print("Add column 'human_exactness' (0 or 1) "
          "then run classifier_performance()")

def classifier_performance(path="annotation_sample.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Run export_annotation_sample() first.")
        return

    if "human_exactness" not in df.columns:
        print("Add 'human_exactness' column first.")
        return

    tp = ((df["exactness_label"]==1) &
          (df["human_exactness"]==1)).sum()
    fp = ((df["exactness_label"]==1) &
          (df["human_exactness"]==0)).sum()
    fn = ((df["exactness_label"]==0) &
          (df["human_exactness"]==1)).sum()
    tn = ((df["exactness_label"]==0) &
          (df["human_exactness"]==0)).sum()

    prec = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0

    print(f"\nClassifier Performance (H2)")
    print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nFalse Negatives (missed exactness-sensitive):")
    fns = df[
        (df["exactness_label"]==0) &
        (df["human_exactness"]==1)
    ]
    for _, row in fns.iterrows():
        print(f"  tool={row['tool_name']} "
              f"conf={row['exactness_confidence']}")
        print(f"  {str(row['raw_preview'])[:150]}")
        print()

def failure_examples(db_path="observations.db", n=5):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT o.task_id, o.tool_name,
               o.raw_content, o.processed_content,
               o.exactness_label,
               o.exactness_confidence,
               s.passed
        FROM observations o
        JOIN sessions s
          ON o.session_id = s.session_id
        WHERE o.condition = 'summarize_all'
          AND o.exactness_label = 1
          AND s.passed = 0
        ORDER BY o.task_id, o.step
        LIMIT ?
    """, conn, params=(n,))
    conn.close()
    return df

def print_report():
    df = load_results()

    print("\n" + "="*60)
    print("H1: PASS RATE BY CONDITION AND TASK TYPE")
    print("(hypothesis: summarize_all degrades on "
          "exactness_sensitive)")
    print("="*60)
    print(pass_rate_table(df).to_string())

    print("\n" + "="*60)
    print("H3: COST BY CONDITION")
    print("(approx_ctx_token_vol = relative proxy only, "
          "not exact)")
    print("="*60)
    print(cost_table(df).to_string())

    print("\n" + "="*60)
    print("FAILURE EXAMPLES")
    print("(summarize_all, exactness-sensitive, failed)")
    print("="*60)
    examples = failure_examples()
    if len(examples) == 0:
        print("No failure examples found.")
    for i, row in examples.iterrows():
        print(f"\n[{i+1}] Task: {row['task_id']} | "
              f"Tool: {row['tool_name']}")
        print(f"Raw:\n  {row['raw_content'][:300]}")
        print(f"Summarized to:\n  {row['processed_content']}")
        print("-"*40)

    print("\n" + "="*60)
    print("H2: CLASSIFIER EVALUATION")
    print("="*60)
    export_annotation_sample()
    classifier_performance()

if __name__ == "__main__":
    print_report()
