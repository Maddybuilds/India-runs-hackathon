import math
import re
from datetime import datetime, timezone

_LOG1P_36 = math.log1p(36)
_LOG1P_50 = math.log1p(50)
_EXPERIENCE_BAND_VARIANCE_2X = 8.0
_ACTIVITY_DECAY_HALFLIFE_DAYS = 180.0
_EXP_TAU = 1.0 / _ACTIVITY_DECAY_HALFLIFE_DAYS
_TODAY = datetime.now(timezone.utc)
_TODAY_YEAR = _TODAY.year
_TODAY_MONTH = _TODAY.month
_TODAY_DAY = _TODAY.day

_AI_TECH_TITLES = frozenset({
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "data engineer", "software engineer",
    "backend engineer", "frontend engineer", "full stack engineer",
    "full-stack engineer", "research engineer", "research scientist",
    "nlp engineer", "computer vision engineer", "deep learning engineer",
    "mlops engineer", "platform engineer", "infrastructure engineer",
    "devops engineer", "site reliability engineer", "sre",
    "cloud engineer", "systems engineer", "applied scientist",
})

_GENERAL_TECH_TITLES = frozenset({
    "full stack developer", "java developer", ".net developer", "mobile developer",
    "qa engineer", "test engineer", "systems engineer",
})

_NONTECH_TITLES = frozenset({
    "marketing manager", "hr manager", "business analyst", "accountant",
    "sales executive", "content writer", "graphic designer",
    "customer support", "operations manager", "project manager",
    "mechanical engineer", "civil engineer", "electrical engineer",
    "financial analyst", "product manager", "consultant",
    "data analyst", "business development", "recruiter",
    "account manager", "client manager", "quality analyst",
    "technical support",
    "seo specialist", "digital marketing", "brand manager",
    "relationship manager", "process analyst", "production engineer",
})

_JD_SKILL_WEIGHTS = {
    "embeddings": 1.0, "embedding": 1.0,
    "retrieval": 1.0, "retrieval augmented generation": 1.0, "rag": 1.0,
    "ranking": 0.8,
    "llm": 1.0, "llms": 1.0, "large language model": 1.0, "large language models": 1.0,
    "python": 0.7, "pytorch": 0.9, "tensorflow": 0.9,
    "vector database": 1.0, "vector databases": 1.0, "milvus": 1.0,
    "pinecone": 1.0, "weaviate": 1.0, "qdrant": 1.0, "chroma": 1.0, "chromadb": 1.0,
    "nlp": 0.9, "natural language processing": 0.9,
    "fine-tuning": 1.0, "fine tuning": 1.0, "fine-tune": 1.0,
    "transformers": 0.9, "huggingface": 0.8, "hugging face": 0.8,
    "langchain": 0.9, "openai": 0.8, "anthropic": 0.8,
    "machine learning": 0.8, "deep learning": 0.9,
    "neural network": 0.8, "neural networks": 0.8,
    "computer vision": 0.8, "cv": 0.7,
    "speech recognition": 0.7, "tts": 0.7, "text to speech": 0.7,
    "gans": 0.7, "generative adversarial": 0.7,
    "diffusion": 0.8, "stable diffusion": 0.8,
    "lora": 0.9, "qlora": 0.9, "peft": 0.9,
    "vector": 0.8,
    "airflow": 0.5, "spark": 0.5, "kafka": 0.4,
    "dbt": 0.4, "snowflake": 0.4, "bigquery": 0.4,
    "scikit-learn": 0.7, "sklearn": 0.7, "xgboost": 0.6, "lightgbm": 0.6,
    "numpy": 0.5, "pandas": 0.5,
    "cuda": 0.7, "gpu": 0.6,
    "mlops": 0.8, "ml ops": 0.8,
    "kubernetes": 0.4, "docker": 0.4,
    "aws": 0.3, "gcp": 0.3, "azure": 0.3,
    "git": 0.3, "ci/cd": 0.3,
    "diffusion models": 0.8,
    "fine-tuning llms": 1.0,
    "hugging face transformers": 0.9,
    "information retrieval": 1.0,
    "information retrieval systems": 1.0,
    "sentence transformers": 0.9,
    "vector search": 0.8,
    "vector representations": 0.8,
    "pgvector": 0.8,
}

_PROFICIENCY_SCORE = {"beginner": 0.25, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}

_CONSULTING_COMPANIES = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "hcltech", "mindtree", "mphasis",
    "ust global", "hexaware", "ltimindtree",
})


def _is_consulting_company(company: str) -> bool:
    c = company.strip().lower()
    return any(brand in c for brand in _CONSULTING_COMPANIES)

_TIER1_INDIA = ("pune", "noida", "delhi", "gurgaon", "hyderabad", "mumbai",
                "bangalore", "bengaluru", "chennai")

_ENG_KEYWORDS = (
    "ml engineer", "software engineer", "data engineer",
    "data scientist", "ai engineer", "machine learning",
    "deep learning", "nlp engineer", "backend engineer",
    "frontend engineer", "full stack", "research scientist",
    "python", "java", "javascript", "typescript", "golang", "rust", "scala",
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "pytorch", "tensorflow", "keras", "scikit-learn",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "data pipeline", "etl", "airflow", "spark", "kafka",
    "microservices", "rest api", "graphql", "grpc",
    "ci/cd", "jenkins", "github actions",
    "sql", "nosql", "data warehousing",
    "computer vision", "nlp", "neural network",
    "architect", "technical lead",
)

_ENG_KW_RE = re.compile("|".join(re.escape(kw) for kw in _ENG_KEYWORDS))

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


def _days_since(s: str) -> int:
    if not s:
        return 180
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10]) if len(s) > 8 else 15
        return max((_TODAY_YEAR - y) * 365 + (_TODAY_MONTH - m) * 30 + (_TODAY_DAY - d), 0)
    except (ValueError, IndexError):
        return 180


def extract_features(candidate: dict, jd_text: str) -> dict:
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    redrob  = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)

    experience_band_fit = math.exp(-((yoe - 7.0) ** 2) / _EXPERIENCE_BAND_VARIANCE_2X)

    current_title = profile.get("current_title", "").strip().lower()
    ai_count = 1 if current_title in _AI_TECH_TITLES else 0
    general_count = 1 if current_title in _GENERAL_TECH_TITLES else 0
    nontech_count = 1 if current_title in _NONTECH_TITLES else 0

    desc_engineering_count = 0
    for h in career:
        t = h.get("title", "").strip().lower()
        if t in _AI_TECH_TITLES:
            ai_count += 1
        elif t in _GENERAL_TECH_TITLES:
            general_count += 1
        elif t in _NONTECH_TITLES:
            nontech_count += 1
        desc = h.get("description", "")
        if desc and _ENG_KW_RE.search(desc):
            desc_engineering_count += 1

    desc_engineering_score = min(1.0, desc_engineering_count / 3.0)

    if ai_count > 0 and nontech_count == 0:
        title_raw = 1.0
    elif ai_count > 0:
        title_raw = 0.75
    elif general_count > 0 and nontech_count == 0:
        title_raw = 0.6
    elif general_count > 0:
        title_raw = 0.45
    elif desc_engineering_score > 0 and nontech_count == 0:
        title_raw = 0.55
    elif desc_engineering_score > 0:
        title_raw = 0.35
    else:
        title_raw = 0.3

    title_engineering_signal = title_raw

    skill_sum = 0.0
    for s in skills:
        weight = _JD_SKILL_WEIGHTS.get(s.get("name", "").strip().lower(), 0.0)
        if weight <= 0.0:
            continue
        prof = _PROFICIENCY_SCORE.get(s.get("proficiency", ""), 0.25)
        dur = s.get("duration_months", 0)
        endo = s.get("endorsements", 0)
        skill_sum += weight * prof * (0.5 * math.log1p(dur) / _LOG1P_36
                                     + 0.5 * math.log1p(endo) / _LOG1P_50)

    skill_quality_score = min(1.0, skill_sum) if skill_sum > 0 else 0.0

    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    will_reloc = redrob.get("willing_to_relocate", False)

    if any(city in loc for city in _TIER1_INDIA):
        location_fit = 1.0
    elif country == "india":
        location_fit = 0.8
    elif will_reloc:
        location_fit = 0.75
    elif country != "india" and not will_reloc:
        location_fit = 0.3
    else:
        location_fit = 0.6

    consulting_only_penalty = 1.0
    if career:
        if all(_is_consulting_company(h.get("company", "")) for h in career):
            consulting_only_penalty = 0.6

    open_to_work = 1.0 if redrob.get("open_to_work_flag", False) else 0.85
    rr = redrob.get("recruiter_response_rate", 0.5)
    icr = redrob.get("interview_completion_rate", 0.5)

    days_inactive = _days_since(redrob.get("last_active_date", ""))

    recency = math.exp(-days_inactive * _EXP_TAU)
    behavioral_multiplier = max(0.25, min(1.0,
        0.20 * open_to_work + 0.25 * rr + 0.25 * icr + 0.30 * recency))

    parts = [profile.get("headline", ""), profile.get("summary", ""), current_title]
    for s in skills:
        parts.append(s.get("name", ""))
    for h in career:
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
    text_blob = " ".join(parts)

    return {
        "years_of_experience":      yoe,
        "experience_band_fit":      experience_band_fit,
        "title_engineering_signal": title_engineering_signal,
        "skill_quality_score":      skill_quality_score,
        "location_fit":             location_fit,
        "consulting_only_penalty":  consulting_only_penalty,
        "behavioral_multiplier":    behavioral_multiplier,
        "text_blob":                text_blob,
        "desc_engineering_score":   desc_engineering_score,
    }
