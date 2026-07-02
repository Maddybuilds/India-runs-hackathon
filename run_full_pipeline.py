"""
End-to-end pipeline showcase — runs the full ranker and produces submission_final.csv
Usage: python run_full_pipeline.py
"""
import os
import sys
import time
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    t0 = time.perf_counter()

    # ── Step 1: Precompute ──
    print("=" * 60)
    print("STEP 1/3 — Precompute features + embeddings")
    print("=" * 60)
    t1 = time.perf_counter()
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from precompute import main as precompute_main
    precompute_main()
    t2 = time.perf_counter()
    print(f"Precompute done in {t2-t1:.1f}s\n")

    # ── Step 2: Rank ──
    print("=" * 60)
    print("STEP 2/3 — Rank candidates")
    print("=" * 60)
    candidates_path = os.path.join(ROOT, "data", "candidates.jsonl")
    submission_path = os.path.join(ROOT, "submission", "submission.csv")

    # Import and run rank.py logic
    import pandas as pd
    import numpy as np

    PARQUET = os.path.join(ROOT, "artifacts", "candidate_features.parquet")
    df = pd.read_parquet(PARQUET, engine="pyarrow")

    mask = ~df["is_honeypot"].values
    semantic = df["semantic_score"].to_numpy(dtype=np.float64)
    title = df["title_engineering_signal"].to_numpy()
    skill = df["skill_quality_score"].to_numpy()
    exp = df["experience_band_fit"].to_numpy()
    loc = df["location_fit"].to_numpy()
    behav = df["behavioral_multiplier"].to_numpy()
    consult = df["consulting_only_penalty"].to_numpy()
    ai_trap = df["ai_recency_trap_penalty"].to_numpy()
    cids = df["candidate_id"].to_numpy()

    score = (0.35 * semantic + 0.25 * title + 0.15 * skill + 0.15 * exp + 0.10 * loc)
    score *= behav * consult * ai_trap
    score[~mask] = -np.inf

    TOP_K = 100
    part_idx = np.argpartition(-score, TOP_K)[:TOP_K]
    sub_order = np.lexsort((cids[part_idx], np.round(-score[part_idx], 4)))
    top_idx = part_idx[sub_order]

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    with open(submission_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i in range(TOP_K):
            w.writerow([cids[top_idx[i]], i + 1, f"{score[top_idx[i]]:.4f}", ""])

    t3 = time.perf_counter()
    print(f"Ranked {TOP_K} candidates in {t3-t2:.1f}s\n")

    # ── Step 3: Generate reasoning ──
    print("=" * 60)
    print("STEP 3/3 — Generate reasoning")
    print("=" * 60)
    from generate_reasoning import main as reasoning_main
    reasoning_main()
    t4 = time.perf_counter()
    print(f"Reasoning done in {t4-t3:.1f}s\n")

    # ── Summary ──
    total = t4 - t0
    print("=" * 60)
    print(f"PIPELINE COMPLETE — {total:.1f}s total")
    print("=" * 60)

    final_path = os.path.join(ROOT, "submission", "submission_final.csv")
    with open(final_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Output: {final_path}")
    print(f"Candidates ranked: {len(rows)}")
    print(f"Score range: {float(rows[0]['score']):.4f} — {float(rows[-1]['score']):.4f}")
    print(f"\nTop 5:")
    for r in rows[:5]:
        print(f"  #{r['rank']} {r['candidate_id']} — {r['score']} — {r['reasoning'][:80]}...")


if __name__ == "__main__":
    main()
