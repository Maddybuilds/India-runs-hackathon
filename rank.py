import argparse
import json
import os
import sys
import time

import pandas as pd
import numpy as np


def main():
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(description="Rank candidates for JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    PARQUET = os.path.join(ROOT, "artifacts", "candidate_features.parquet")

    df = pd.read_parquet(PARQUET, engine="pyarrow")

    if os.path.getsize(args.candidates) < 50_000_000:
        live_ids = set()
        with open(args.candidates, encoding="utf-8") as _fh:
            for _line in _fh:
                if not _line.strip():
                    continue
                live_ids.add(json.loads(_line)["candidate_id"])
        pq_ids = set(df["candidate_id"])
        missing = pq_ids - live_ids
        extra = live_ids - pq_ids
        if missing:
            print(f"[rank] WARNING: {len(missing)} IDs in parquet not in live file")
        if extra:
            print(f"[rank] WARNING: {len(extra)} IDs in live file not in parquet")

    n_before = len(df)
    mask = ~df["is_honeypot"].values
    n_filtered = n_before - mask.sum()
    if n_filtered:
        print(f"[rank] Filtered {n_filtered} honeypots")

    semantic = df["semantic_score"].to_numpy(dtype=np.float64, copy=False)
    title = df["title_engineering_signal"].to_numpy(copy=False)
    skill = df["skill_quality_score"].to_numpy(copy=False)
    exp = df["experience_band_fit"].to_numpy(copy=False)
    loc = df["location_fit"].to_numpy(copy=False)
    behav = df["behavioral_multiplier"].to_numpy(copy=False)
    consult = df["consulting_only_penalty"].to_numpy(copy=False)
    ai_trap = df["ai_recency_trap_penalty"].to_numpy(copy=False)
    cids = df["candidate_id"].to_numpy(copy=False)

    score = (0.35 * semantic + 0.25 * title + 0.15 * skill + 0.15 * exp + 0.10 * loc)
    score *= behav
    score *= consult
    score *= ai_trap

    score[~mask] = -np.inf

    TOP_K = 100
    part_idx = np.argpartition(-score, TOP_K)[:TOP_K]
    sub_order = np.lexsort((cids[part_idx], np.round(-score[part_idx], 4)))
    top_idx = part_idx[sub_order]

    top_cids = cids[top_idx]
    top_scores = score[top_idx]
    rounded_scores = np.round(top_scores, 4)
    ranks = np.arange(1, 101, dtype=np.int32)

    assert np.all(rounded_scores[:-1] >= rounded_scores[1:]), "Scores not non-increasing!"
    assert np.array_equal(ranks, np.arange(1, 101)), "Ranks not 1-100!"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import csv
    with open(args.out, "w", encoding="utf-8", newline="") as _f:
        _w = csv.writer(_f)
        _w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i in range(TOP_K):
            _w.writerow([top_cids[i], ranks[i], f"{top_scores[i]:.4f}", ""])

    print("\n" + "=" * 60)
    print("[audit] Top-100 summary")
    print("=" * 60)
    
    top_title_scores = df.loc[top_idx, "title_engineering_signal"].values
    print(f"[audit] Top-100 title_engineering_signal: min={top_title_scores.min():.2f} mean={top_title_scores.mean():.2f}")
    
    top_honeypots = df.loc[top_idx, "is_honeypot"].values.sum()
    print(f"[audit] Top-100 honeypot count: {top_honeypots} (should be 0 — already filtered)")
    
    top_consulting = (df.loc[top_idx, "consulting_only_penalty"].values < 1.0).sum()
    print(f"[audit] Top-100 consulting_only_penalty < 1.0 count: {top_consulting}")
    print("=" * 60)

    elapsed = time.perf_counter() - t0
    print(f"[rank] Wrote {args.out} ({TOP_K} rows)")
    print(f"[rank] Elapsed: {elapsed:.2f}s")
    assert elapsed < 300, f"TIMEOUT: {elapsed:.1f}s > 300s limit!"


if __name__ == "__main__":
    main()
