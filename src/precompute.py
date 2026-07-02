import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

try:
    import orjson
    def loads(line): return orjson.loads(line)
except ImportError:
    import json
    def loads(line): return json.loads(line)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from features import extract_features, _ENG_KEYWORDS
from honeypot_rules import is_honeypot, ai_recency_trap_penalty

_ENG_KW_RE = re.compile("|".join(re.escape(kw) for kw in _ENG_KEYWORDS))

WORKERS = min(mp.cpu_count(), 8)
CHUNK_SIZE = 20000


def _process_chunk(args):
    chunk, jd_text = args
    chunk_rows = []
    chunk_blobs = []
    for raw in chunk:
        cand = loads(raw)
        cid = cand["candidate_id"]
        feats = extract_features(cand, jd_text)
        text_blob = feats.pop("text_blob")
        chunk_rows.append((
            cid,
            feats["years_of_experience"],
            feats["experience_band_fit"],
            feats["title_engineering_signal"],
            feats["skill_quality_score"],
            feats["location_fit"],
            feats["consulting_only_penalty"],
            feats["behavioral_multiplier"],
            is_honeypot(cand),
            ai_recency_trap_penalty(cand),
            feats["desc_engineering_score"],
            cand.get("profile", {}).get("current_title", ""),
        ))
        chunk_blobs.append(text_blob)
    return chunk_rows, chunk_blobs


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

JD_DOCX = os.path.join(DATA_DIR, "job_description.docx")
JD_MD = os.path.join(DATA_DIR, "job_description.md")
CANDIDATES_JSONL = os.path.join(DATA_DIR, "candidates.jsonl")

PARQUET_OUT = os.path.join(ARTIFACTS_DIR, "candidate_features.parquet")
TFIDF_OUT = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
SVD_OUT = os.path.join(ARTIFACTS_DIR, "svd_model.joblib")



def load_jd_text() -> str:
    if os.path.exists(JD_MD):
        with open(JD_MD, encoding="utf-8") as f:
            return f.read()
    if os.path.exists(JD_DOCX):
        try:
            import docx
            doc = docx.Document(JD_DOCX)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            pass
    raise FileNotFoundError("Cannot read job description")



def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    t0 = time.perf_counter()

    jd_text = load_jd_text()
    print(f"[precompute] JD loaded ({len(jd_text)} chars)")

    raw_lines = []
    with open(CANDIDATES_JSONL, "rb") as fh:
        for line in fh:
            if line.strip():
                raw_lines.append(line)
    print(f"[precompute] Loaded {len(raw_lines)} raw lines ({time.perf_counter()-t0:.1f}s)")

    n = len(raw_lines)
    chunks = [raw_lines[i:i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]
    rows = []
    text_blobs = []

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_process_chunk, (c, jd_text)): i for i, c in enumerate(chunks)}
        done = 0
        for future in as_completed(futures):
            chunk_rows, chunk_blobs = future.result()
            rows.extend(chunk_rows)
            text_blobs.extend(chunk_blobs)
            done += 1
            print(f"[precompute] Chunk {done}/{len(chunks)} done  ({time.perf_counter()-t0:.1f}s)")

    elapsed_feat = time.perf_counter() - t0
    print(f"[precompute] Features done: {len(rows)} candidates  ({elapsed_feat:.1f}s)")

    COLS = [
        "candidate_id", "years_of_experience", "experience_band_fit",
        "title_engineering_signal", "skill_quality_score", "location_fit",
        "consulting_only_penalty", "behavioral_multiplier",
        "is_honeypot", "ai_recency_trap_penalty",
        "desc_engineering_score", "current_title",
    ]
    df = pd.DataFrame.from_records(rows, columns=COLS)

    text_blobs.append(jd_text)
    assert len(text_blobs) == len(rows) + 1, "JD text not appended correctly"
    tfidf = TfidfVectorizer(
        ngram_range=(1, 1), max_features=15000,
        stop_words="english", sublinear_tf=True, dtype=np.float32,
    )
    tfidf_matrix = tfidf.fit_transform(text_blobs)
    jd_vec = tfidf_matrix[-1]
    cand_matrix = tfidf_matrix[:-1]
    print(f"[precompute] TF-IDF done: {cand_matrix.shape}")

    n_components = min(100, cand_matrix.shape[1] - 1, cand_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    cand_svd = svd.fit_transform(cand_matrix)
    jd_svd = svd.transform(jd_vec)
    print(f"[precompute] SVD done: explained={svd.explained_variance_ratio_.sum():.3f}")

    df["semantic_score"] = np.round(cosine_similarity(cand_svd, jd_svd).ravel(), 6)

    REF_STRONG_AI = [
        "Machine Learning Engineer", "AI Engineer", "NLP Engineer",
        "Data Scientist", "Computer Vision Engineer", "AI Specialist",
        "Junior ML Engineer", "AI Research Engineer",
        "Senior Software Engineer (ML)", "Recommendation Systems Engineer",
    ]
    REF_GENERAL_ENG = [
        "Software Engineer", "Full Stack Developer", "Java Developer",
        "Backend Engineer", "Frontend Engineer", "DevOps Engineer",
        "Cloud Engineer", "Data Engineer", "QA Engineer", ".NET Developer",
    ]
    REF_NONTECH = [
        "Marketing Manager", "HR Manager", "Business Analyst",
        "Accountant", "Sales Executive", "Content Writer",
        "Project Manager", "Operations Manager", "Customer Support",
        "Graphic Designer",
    ]

    all_ref = REF_STRONG_AI + REF_GENERAL_ENG + REF_NONTECH
    ref_tfidf = tfidf.transform(all_ref)
    ref_svd = svd.transform(ref_tfidf)

    n_strong = len(REF_STRONG_AI)
    n_general = len(REF_GENERAL_ENG)
    n_nontech = len(REF_NONTECH)

    strong_ref = ref_svd[:n_strong]
    general_ref = ref_svd[n_strong:n_strong + n_general]
    nontech_ref = ref_svd[n_strong + n_general:]

    titles = df["current_title"].tolist()
    title_tfidf = tfidf.transform(titles)
    title_svd = svd.transform(title_tfidf)

    sim_strong = cosine_similarity(title_svd, strong_ref).max(axis=1)
    sim_general = cosine_similarity(title_svd, general_ref).max(axis=1)
    sim_nontech = cosine_similarity(title_svd, nontech_ref).max(axis=1)

    eps = 1e-6
    embedding_score = (sim_strong + 0.5 * sim_general) / (sim_strong + sim_general + sim_nontech + eps)
    embedding_score = np.clip(embedding_score, 0.0, 1.0)

    desc_signal = df["desc_engineering_score"].values
    title_blended = 0.70 * embedding_score + 0.30 * desc_signal

    old_tier_values = df["title_engineering_signal"].values.copy()
    df["title_engineering_signal"] = np.round(title_blended, 6)

    print("\n" + "=" * 60)
    print("[audit] title_engineering_signal: tier-based vs embedding-based")
    print("=" * 60)

    correlation = np.corrcoef(old_tier_values, title_blended)[0, 1]
    print(f"[audit] Correlation between old and new: {correlation:.4f}")

    diff = np.abs(title_blended - old_tier_values)
    top20_idx = np.argsort(diff)[-20:][::-1]
    print(f"[audit] Top 20 candidates with largest score change:")
    for i, idx in enumerate(top20_idx, 1):
        old_v = old_tier_values[idx]
        new_v = title_blended[idx]
        title_str = titles[idx]
        print(f"  {i:2d}. {title_str!r}: {old_v:.4f} -> {new_v:.4f} (delta={new_v-old_v:+.4f})")

    print(f"\n[audit] New title_engineering_signal distribution:")
    new_dist = df["title_engineering_signal"]
    print(f"      min={new_dist.min():.4f} max={new_dist.max():.4f} mean={new_dist.mean():.4f} std={new_dist.std():.4f}")
    print(f"      [0.0-0.2): {(new_dist < 0.2).sum()}")
    print(f"      [0.2-0.4): {((new_dist >= 0.2) & (new_dist < 0.4)).sum()}")
    print(f"      [0.4-0.6): {((new_dist >= 0.4) & (new_dist < 0.6)).sum()}")
    print(f"      [0.6-0.8): {((new_dist >= 0.6) & (new_dist < 0.8)).sum()}")
    print(f"      [0.8-1.0]: {(new_dist >= 0.8).sum()}")
    print("=" * 60)

    desc_dist = df["desc_engineering_score"]
    total = len(df)
    print(f"\n[audit] desc_engineering_score distribution:")
    print(f"      min={desc_dist.min():.4f} max={desc_dist.max():.4f} mean={desc_dist.mean():.4f}")
    print(f"      [0]: {(desc_dist == 0).sum()} ({(desc_dist == 0).sum()/total*100:.1f}%)")
    print(f"      (0,0.34]: {((desc_dist > 0) & (desc_dist <= 0.34)).sum()}")
    print(f"      (0.34,0.67]: {((desc_dist > 0.34) & (desc_dist <= 0.67)).sum()}")
    print(f"      (0.67,1.0]: {((desc_dist > 0.67) & (desc_dist <= 1.0)).sum()}")
    print("=" * 60)

    df.drop(columns=["desc_engineering_score", "current_title"], inplace=True)

    df.to_parquet(PARQUET_OUT, index=False)
    joblib.dump(tfidf, TFIDF_OUT, compress=3)
    joblib.dump(svd, SVD_OUT, compress=3)

    print("\n" + "=" * 60)
    print("[audit] Precompute summary")
    print("=" * 60)
    
    total = len(df)
    honeypot_count = df["is_honeypot"].sum()
    print(f"[audit] is_honeypot fired on: {honeypot_count} / {total}")
    
    ai_trap_count = (df["ai_recency_trap_penalty"] < 1.0).sum()
    print(f"[audit] ai_recency_trap_penalty < 1.0 on: {ai_trap_count} / {total}")
    
    consulting_count = (df["consulting_only_penalty"] < 1.0).sum()
    print(f"[audit] consulting_only_penalty < 1.0 on: {consulting_count} / {total}")
    
    print(f"[audit] title_engineering_signal distribution:")
    title_dist = df["title_engineering_signal"].value_counts().sort_index()
    for score, count in title_dist.items():
        print(f"      {score}: {count} ({count/total*100:.1f}%)")
    
    print(f"[audit] skill_quality_score: min={df['skill_quality_score'].min():.4f} max={df['skill_quality_score'].max():.4f} mean={df['skill_quality_score'].mean():.4f}")
    print("=" * 60)

    elapsed = time.perf_counter() - t0
    print(f"[precompute] Total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
