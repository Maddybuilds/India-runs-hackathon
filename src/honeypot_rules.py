from datetime import datetime, timezone
import re

_ADVANCED_PROF = frozenset({"advanced", "expert"})

_AI_KEYWORDS = (
    "llm", "llms", "large language model", "langchain", "llamaindex",
    "openai", "anthropic", "rag", "retrieval augmented generation",
    "vector database", "vector databases", "embeddings", "embedding",
    "fine-tuning", "fine tuning", "fine-tune", "prompt engineering",
    "transformers", "huggingface", "hugging face", "chatgpt", "gpt",
)

_ML_KEYWORDS = (
    "machine learning", "ml engineer", "data scientist", "data engineer",
    "deep learning", "nlp", "natural language processing",
    "computer vision", "mlops", "ai engineer", "research scientist",
    "research engineer", "applied scientist", "feature engineering",
    "model training", "model deployment", "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "xgboost",
)

_BACKEND_KEYWORDS = (
    "software engineer", "backend engineer", "frontend engineer",
    "full stack", "fullstack", "platform engineer", "infrastructure engineer",
    "devops", "site reliability", "sre", "cloud engineer", "systems engineer",
    "data pipeline", "data warehouse", "etl", "airflow", "spark", "kafka",
)

_AI_KW_RE = re.compile("|".join(re.escape(kw) for kw in _AI_KEYWORDS))
_ML_KW_RE = re.compile("|".join(re.escape(kw) for kw in _ML_KEYWORDS))
_BE_KW_RE = re.compile("|".join(re.escape(kw) for kw in _BACKEND_KEYWORDS))

_TODAY = datetime.now(timezone.utc)


def _parse_date_fast(s: str):
    if not s:
        return None
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10]) if len(s) > 8 else 1
        return datetime(y, m, d, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def _months_between(d1, d2):
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    for s in skills:
        if s.get("proficiency", "") in _ADVANCED_PROF and s.get("duration_months", 0) <= 2:
            return True

    total_months = 0
    for h in career:
        total_months += h.get("duration_months", 0)
    if total_months / 12.0 > profile.get("years_of_experience", 0) + 3:
        return True

    for h in career:
        end_str = h.get("end_date")
        if not end_str:
            continue
        start = _parse_date_fast(h.get("start_date", ""))
        end = _parse_date_fast(end_str)
        if start and end and end < start:
            return True
        if start and end:
            actual = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(h.get("duration_months", 0) - actual) > 2:
                return True

    return False


def ai_recency_trap_penalty(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    llm_durs = []
    for s in skills:
        sname = s.get("name", "").lower()
        if _AI_KW_RE.search(sname):
            llm_durs.append(s.get("duration_months", 0))

    if not llm_durs:
        return 1.0

    for d in llm_durs:
        if d >= 12:
            return 1.0

    for h in career:
        start = _parse_date_fast(h.get("start_date", ""))
        if not start:
            continue
        if _months_between(start, _TODAY) <= 12:
            continue
        title_lower = h.get("title", "").lower()
        desc_lower = h.get("description", "").lower()
        if _ML_KW_RE.search(title_lower) or _ML_KW_RE.search(desc_lower):
            return 1.0
        if _BE_KW_RE.search(title_lower) or _BE_KW_RE.search(desc_lower):
            return 1.0

    return 0.5
