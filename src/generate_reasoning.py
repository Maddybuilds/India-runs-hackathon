import csv
import hashlib
import json
import os
import sys
import time
import io

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES_JSONL = os.path.join(ROOT, "data", "candidates.jsonl")
SUBMISSION_CSV = os.path.join(ROOT, "submission", "submission.csv")
SUBMISSION_FINAL_CSV = os.path.join(ROOT, "submission", "submission_final.csv")

CORE_SKILLS = {"embeddings", "embedding", "sentence-transformers", "bge", "e5",
               "rag", "retrieval augmented generation", "retrieval", "vector database",
               "vector databases", "pinecone", "weaviate", "qdrant", "milvus",
               "faiss", "opensearch", "elasticsearch", "chroma", "pgvector"}
EVAL_SKILLS = {"ndcg", "mrr", "map", "ranking", "evaluation", "a/b testing"}
FINE_TUNE_SKILLS = {"lora", "qlora", "peft", "fine-tuning", "fine tuning", "transformers"}
LR_SKILLS = {"xgboost", "lightgbm", "learning-to-rank", "ranking"}
PY_SKILLS = {"python", "pytorch", "tensorflow", "scikit-learn", "sklearn"}

CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
              "tech mahindra", "hcl", "hcltech", "mindtree"}
TIER1 = {"pune", "noida", "delhi", "gurgaon", "hyderabad", "mumbai",
         "bangalore", "bengaluru", "chennai"}

TECH_TITLES = {"ml engineer", "data scientist", "data engineer", "software engineer",
               "backend engineer", "frontend engineer", "ai engineer", "nlp engineer",
               "research scientist", "research engineer", "mlops engineer",
               "platform engineer", "systems engineer", "applied scientist"}


def _variant(candidate_id: str, n: int) -> int:
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16) % n


OPENERS_STRONG = ["Strong fit at #{r}:", "Top contender at #{r}:", "Excellent match at #{r}:"]
OPENERS_GOOD = ["Good fit at #{r}:", "Solid candidate at #{r}:", "Viable match at #{r}:"]
OPENERS_PARTIAL = ["Partial fit at #{r}:", "Moderate fit at #{r}:", "Some alignment at #{r}:"]
OPENERS_WEAK = ["Weak fit at #{r}:", "Limited match at #{r}:", "Marginal fit at #{r}:"]
OPENERS_BORDERLINE = ["Borderline at #{r}:", "Edge case at #{r}:", "Uncertain fit at #{r}:"]

CONNECTIVES_SKILLS = [
    "with hands-on {skills}",
    "showing experience in {skills}",
    "demonstrating {skills} expertise",
]

CONNECTIVES_LOCATION = [
    "based in {loc}",
    "located in {loc}",
    "from {loc}",
]


def get_skill_names(skills):
    return {s.get("name", "").lower().strip() for s in skills}


def has_skill_overlap(candidate_skills, target_set):
    return candidate_skills & target_set


def count_consulting(career):
    return sum(1 for h in career if h.get("company", "").lower().strip() in CONSULTING)


def get_titles(career, current_title):
    titles = [current_title.lower()]
    titles.extend(h.get("title", "").lower() for h in career)
    return titles


def generate_reasoning(candidate, rank):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})
    edu = candidate.get("education", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    skill_names = get_skill_names(skills)
    cid = candidate.get("candidate_id", "")

    parts = []

    parts.append(f"{yoe} years as {title} at {company}")

    core_hit = skill_names & CORE_SKILLS
    eval_hit = skill_names & EVAL_SKILLS
    ft_hit = skill_names & FINE_TUNE_SKILLS
    lr_hit = skill_names & LR_SKILLS
    py_hit = skill_names & PY_SKILLS

    v = _variant(cid, len(CONNECTIVES_SKILLS))
    if core_hit:
        skills_str = ", ".join(sorted(core_hit)[:4])
        parts.append(CONNECTIVES_SKILLS[v].format(skills=skills_str))
    if eval_hit:
        parts.append(f"eval experience ({', '.join(sorted(eval_hit)[:3])})")
    if ft_hit:
        parts.append(f"fine-tuning ({', '.join(sorted(ft_hit)[:3])})")
    if lr_hit:
        parts.append(f"learning-to-rank ({', '.join(sorted(lr_hit)[:2])})")

    n_consulting = count_consulting(career)
    n_total = len(career)
    if n_consulting == n_total and n_total > 0:
        parts.append("solely at consulting firms")
    elif n_consulting > 0:
        parts.append(f"mixed background ({n_consulting}/{n_total} consulting)")

    loc_lower = location.lower()
    v_loc = _variant(cid, len(CONNECTIVES_LOCATION))
    if any(c in loc_lower for c in TIER1):
        loc_str = CONNECTIVES_LOCATION[v_loc].format(loc=f"{location} (Tier-1 India)")
        parts.append(loc_str)
    elif country.lower() == "india":
        loc_str = CONNECTIVES_LOCATION[v_loc].format(loc=f"{location}, India")
        parts.append(loc_str)
    elif redrob.get("willing_to_relocate"):
        loc_str = CONNECTIVES_LOCATION[v_loc].format(loc=f"{location} but willing to relocate")
        parts.append(loc_str)
    else:
        loc_str = CONNECTIVES_LOCATION[v_loc].format(loc=f"{location} ({country})")
        parts.append(loc_str)

    otw = redrob.get("open_to_work_flag", False)
    rr = redrob.get("recruiter_response_rate", 0)
    la = redrob.get("last_active_date", "")
    notice = redrob.get("notice_period_days", 0)

    signals = []
    if otw:
        signals.append("open to work")
    if rr and rr < 0.3:
        signals.append(f"low recruiter response ({rr:.0%})")
    if notice and notice > 90:
        signals.append(f"{notice}-day notice")
    if signals:
        parts.append("; ".join(signals))

    if rank <= 10:
        opener = OPENERS_STRONG[_variant(cid, len(OPENERS_STRONG))].format(r=rank)
    elif rank <= 30:
        opener = OPENERS_GOOD[_variant(cid, len(OPENERS_GOOD))].format(r=rank)
    elif rank <= 60:
        opener = OPENERS_PARTIAL[_variant(cid, len(OPENERS_PARTIAL))].format(r=rank)
    elif rank <= 80:
        opener = OPENERS_WEAK[_variant(cid, len(OPENERS_WEAK))].format(r=rank)
    else:
        opener = OPENERS_BORDERLINE[_variant(cid, len(OPENERS_BORDERLINE))].format(r=rank)

    body = ", ".join(parts)

    gaps = []
    if not core_hit:
        gaps.append("no direct retrieval/vector-DB experience")
    if not py_hit:
        gaps.append("limited Python depth")
    if yoe < 5:
        gaps.append(f"below 5yr experience floor ({yoe}yr)")
    if yoe > 9:
        gaps.append(f"exceeds 5-9yr band ({yoe}yr)")

    if gaps:
        body += f". Gaps: {', '.join(gaps[:2])}"

    return f"{opener} {body}."


def main():
    t0 = time.perf_counter()

    with open(SUBMISSION_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    target_ids = {row["candidate_id"] for row in rows}
    print(f"[reasoning] {len(rows)} candidates")

    candidates = {}
    with open(CANDIDATES_JSONL, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            cand = _loads(line)
            cid = cand["candidate_id"]
            if cid in target_ids:
                candidates[cid] = cand
                if len(candidates) == len(target_ids):
                    break

    reasoning_map = {}
    for row in rows:
        cid = row["candidate_id"]
        cand = candidates.get(cid)
        if cand:
            reasoning_map[cid] = generate_reasoning(cand, int(row["rank"]))
        else:
            reasoning_map[cid] = ""

    with open(SUBMISSION_FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": row["score"],
                "reasoning": reasoning_map.get(row["candidate_id"], ""),
            })

    filled = sum(1 for v in reasoning_map.values() if v)
    elapsed = time.perf_counter() - t0
    print(f"[reasoning] Wrote {SUBMISSION_FINAL_CSV} ({filled}/{len(rows)} filled) in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
