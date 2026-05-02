import argparse
import json
import os
import shutil
from agent import run_agent, init_db

REPLICATES = 3
CONDITIONS = ["full_exact", "summarize_all", "exactness_aware"]

PILOT_TASKS = {"task_01", "task_05"}  # one exactness-sensitive, one not

def load_tasks(tasks_dir="tasks") -> list[dict]:
    tasks = []
    for task_id in sorted(os.listdir(tasks_dir)):
        task_dir = os.path.join(tasks_dir, task_id)
        meta_path = os.path.join(task_dir, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            import json as j
            meta = j.load(f)
        tasks.append({
            "id": task_id,
            "code_file": os.path.join(task_dir, "buggy.py"),
            "test_file": os.path.join(task_dir, "test_buggy.py"),
            "type": meta["type"]
        })
    return tasks

def reset_tasks(tasks_dir="tasks"):
    for task_id in os.listdir(tasks_dir):
        task_dir = os.path.join(tasks_dir, task_id)
        original = os.path.join(task_dir, "buggy_original.py")
        buggy = os.path.join(task_dir, "buggy.py")
        if os.path.exists(original):
            shutil.copy(original, buggy)

def run_all(tasks_dir="tasks", pilot=False, replicates=None, pressure=False):
    conn = init_db()
    tasks = load_tasks(tasks_dir)
    if pilot:
        tasks = [t for t in tasks if t["id"] in PILOT_TASKS]
    if replicates is None:
        replicates = 1 if pilot else REPLICATES
    all_results = []

    pressure_tag = "+pressure" if pressure else ""
    total = len(tasks) * len(CONDITIONS) * replicates
    mode = "PILOT" if pilot else "FULL"
    print(f"[{mode}{pressure_tag}] Running {total} sessions total")
    print(f"({len(tasks)} tasks x {len(CONDITIONS)} conditions "
          f"x {replicates} replicates)")

    for condition in CONDITIONS:
        print(f"\n{'='*50}\nCondition: {condition}{pressure_tag}\n{'='*50}")
        for task in tasks:
            for rep in range(replicates):
                reset_tasks(tasks_dir)
                label = (f"  {task['id']} "
                         f"({task['type']}) "
                         f"rep {rep+1}/{replicates}")
                print(f"{label}... ", end="", flush=True)
                try:
                    result = run_agent(
                        task, condition, rep, conn, pressure=pressure
                    )
                    all_results.append(result)
                    status = "PASS" if result["passed"] else "FAIL"
                    print(
                        f"{status} | "
                        f"ctx~{result['approx_ctx_token_volume']} "
                        f"billed={result['total_billed_tokens']} "
                        f"lat={result['latency_seconds']}s"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")

    existing = []
    if os.path.exists("results.json"):
        with open("results.json") as f:
            existing = json.load(f)
    with open("results.json", "w") as f:
        json.dump(existing + all_results, f, indent=2)

    conn.close()
    print(f"\nDone. {len(all_results)} sessions logged.")
    print("Results: results.json | Archive: observations.db")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="2 tasks x 1 replicate for cost calibration")
    parser.add_argument("--replicates", type=int, default=None,
                        help=f"Override replicate count (default {REPLICATES})")
    parser.add_argument("--pressure", action="store_true",
                        help="Pre-fill context with synthetic noise observations")
    args = parser.parse_args()
    run_all(pilot=args.pilot, replicates=args.replicates, pressure=args.pressure)
