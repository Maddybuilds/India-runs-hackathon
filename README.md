# Redrob Hackathon  Team OffGridDev

Embedding-based semantic ranker with rule-based disqualifiers for the
Intelligent Candidate Discovery & Ranking Challenge.

## Demo / Showcase

https://github.com/Maddybuilds/India-runs-hackathon/assets/demo-showcase.mp4

> **How it works:** Watch the full walkthrough — from data ingestion to ranked candidates with reasoning.

<details>
<summary>Alternatively, <a href="demo-showcase.mp4">download the video directly</a></summary>
</details>

## Setup

```bash
pip install -r requirements.txt
```

Place these files in `data/`:
- `candidates.jsonl`
- `candidate_schema.json`
- `sample_candidates.json`
- `job_description.docx` (or `job_description.md`)

## Reproduce Submission

### Full pipeline (precompute + rank + reasoning):
```bash
python run_full_pipeline.py
```

### Or step by step:
```bash
python src/precompute.py
python rank.py --candidates ./data/candidates.jsonl --out ./submission/submission.csv
python src/generate_reasoning.py
```

Output: `submission/submission_final.csv` (100 ranked candidates)

## Architecture

- **Precompute**: TF-IDF + SVD embeddings for all 100K candidates, cosine similarity to JD
- **Ranking formula**: `(0.35×semantic + 0.25×title + 0.15×skill + 0.15×exp + 0.10×loc) × behav × consult × ai_trap`
- **Honeypots**: Filtered out (advanced skill ≤2 months, career anomalies, date inconsistencies)
- **Reasoning**: Template-based from verified candidate facts — zero LLM calls

## Compute

- Runtime: <0.1s on CPU (ranking step)
- Precompute: ~1 min (offline, one-time)
- No GPU, no network during ranking
