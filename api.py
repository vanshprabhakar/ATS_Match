"""FastAPI backend for AI-powered Resume-JD matching."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger("resume_match_api")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.\,\-\+\#\/]")
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9\+\#\.\-]{1,30}")

SKILL_LEXICON = {
    "python",
    "java",
    "c++",
    "sql",
    "nosql",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "fastapi",
    "django",
    "flask",
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "nlp",
    "llm",
    "transformers",
    "git",
    "linux",
    "spark",
    "hadoop",
    "tableau",
    "powerbi",
    "excel",
    "rest",
    "microservices",
    "agile",
    "jira",
}


class PredictRequest(BaseModel):
    """Incoming prediction payload."""

    resume_text: str = Field(..., min_length=20, description="Candidate resume text")
    jd_text: str = Field(..., min_length=20, description="Job description text")


class PredictResponse(BaseModel):
    """Prediction response payload."""

    match_probability: float
    predicted_label: int
    confidence_band: str
    cosine_similarity: float
    matched_skills: list[str]
    missing_skills: list[str]


class RankBatchRequest(BaseModel):
    """Batch ranking payload for one JD and many candidate resumes."""

    jd_text: str = Field(..., min_length=20, description="Job description text")
    candidate_resumes: list[str] = Field(
        ...,
        min_length=1,
        description="Candidate resume texts to rank against the JD.",
    )
    candidate_names: list[str] | None = Field(
        default=None,
        description="Optional display names aligned by index with candidate_resumes.",
    )


class RankedCandidate(BaseModel):
    """Ranked candidate response row."""

    rank: int
    candidate_index: int
    candidate_name: str
    match_probability: float
    predicted_label: int
    confidence_band: str
    cosine_similarity: float
    matched_skills: list[str]
    missing_skills: list[str]
    resume_text: str


class RankBatchResponse(BaseModel):
    """Batch ranking response."""

    jd_text: str
    total_candidates: int
    ranked_candidates: list[RankedCandidate]


@dataclass
class AppState:
    """In-memory artifact handles."""

    embedder: SentenceTransformer
    model: object


def configure_logging() -> None:
    """Configure API logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clean_text(text: str) -> str:
    """Normalize text consistently with training preprocessing."""
    normalized = text.strip().lower()
    normalized = NON_TEXT_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def build_pair_features(resume_emb: np.ndarray, jd_emb: np.ndarray) -> tuple[np.ndarray, float]:
    """Build engineered vector features from embedding pairs."""
    resume_emb = resume_emb.astype(np.float32)
    jd_emb = jd_emb.astype(np.float32)
    dot_product = float(np.sum(resume_emb * jd_emb))
    cosine_similarity = dot_product  # embeddings are normalized
    l2_distance = float(np.linalg.norm(resume_emb - jd_emb))
    abs_diff = np.abs(resume_emb - jd_emb)
    hadamard = resume_emb * jd_emb

    features = np.hstack(
        [
            resume_emb,
            jd_emb,
            abs_diff,
            hadamard,
            np.array([cosine_similarity], dtype=np.float32),
            np.array([l2_distance], dtype=np.float32),
        ]
    ).reshape(1, -1)
    return features, cosine_similarity


def build_pair_features_batch(resume_embs: np.ndarray, jd_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build engineered vector features for many resumes against one JD embedding."""
    jd_matrix = np.repeat(jd_emb.reshape(1, -1), repeats=resume_embs.shape[0], axis=0).astype(np.float32)
    resume_matrix = resume_embs.astype(np.float32)
    cosine_sim = np.sum(resume_matrix * jd_matrix, axis=1, keepdims=True)
    l2_distance = np.linalg.norm(resume_matrix - jd_matrix, axis=1, keepdims=True)
    abs_diff = np.abs(resume_matrix - jd_matrix)
    hadamard = resume_matrix * jd_matrix
    features = np.hstack([resume_matrix, jd_matrix, abs_diff, hadamard, cosine_sim, l2_distance]).astype(np.float32)
    return features, cosine_sim.reshape(-1)


def extract_skills(text: str) -> set[str]:
    """Extract normalized skills using a curated lexicon and token matching."""
    tokens = {token.lower() for token in TOKEN_PATTERN.findall(text.lower())}
    return {skill for skill in SKILL_LEXICON if skill in tokens or skill in text.lower()}


def confidence_band(probability: float) -> str:
    """Convert probability into interpretable confidence ranges."""
    if probability >= 0.80:
        return "High"
    if probability >= 0.60:
        return "Medium"
    return "Low"


def load_state() -> AppState:
    """Load embedding model and classifier artifacts from disk."""
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    model_path = artifacts_dir / "hist_gbc_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run src/02_train_model.py first."
        )

    sentence_model_name = os.getenv("SENTENCE_MODEL_NAME", "all-MiniLM-L6-v2")
    embedder = SentenceTransformer(sentence_model_name)
    model = joblib.load(model_path)
    LOGGER.info("Artifacts loaded from %s", artifacts_dir)
    return AppState(embedder=embedder, model=model)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load artifacts during API startup and release state on shutdown."""
    configure_logging()
    app.state.runtime = load_state()
    try:
        yield
    finally:
        app.state.runtime = None


app = FastAPI(
    title="AI Resume-JD Matching API",
    version="1.0.0",
    description="Enterprise-grade decision-support API for Resume-JD semantic alignment.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight liveness endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Generate match score, class prediction, and skill-gap insights."""
    runtime: AppState | None = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    try:
        resume_text = clean_text(payload.resume_text)
        jd_text = clean_text(payload.jd_text)

        resume_emb = runtime.embedder.encode(
            resume_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        jd_emb = runtime.embedder.encode(
            jd_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        features, cosine_sim = build_pair_features(resume_emb, jd_emb)
        probability = float(runtime.model.predict_proba(features)[0][1])
        predicted_label = int(runtime.model.predict(features)[0])

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        matched_skills = sorted(resume_skills.intersection(jd_skills))
        missing_skills = sorted(jd_skills.difference(resume_skills))

        return PredictResponse(
            match_probability=probability,
            predicted_label=predicted_label,
            confidence_band=confidence_band(probability),
            cosine_similarity=cosine_sim,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failure: {exc}") from exc


@app.post("/rank_batch", response_model=RankBatchResponse)
def rank_batch(payload: RankBatchRequest) -> RankBatchResponse:
    """Rank multiple resumes against one JD and return a sorted leaderboard."""
    runtime: AppState | None = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    if payload.candidate_names is not None and len(payload.candidate_names) != len(payload.candidate_resumes):
        raise HTTPException(
            status_code=422,
            detail="candidate_names length must match candidate_resumes length.",
        )

    try:
        jd_text = clean_text(payload.jd_text)
        if len(jd_text) < 20:
            raise HTTPException(status_code=422, detail="JD text is too short after preprocessing.")

        cleaned_resumes: list[str] = []
        for idx, resume in enumerate(payload.candidate_resumes):
            cleaned = clean_text(resume)
            if len(cleaned) < 20:
                raise HTTPException(
                    status_code=422,
                    detail=f"Candidate resume at index {idx} is too short after preprocessing.",
                )
            cleaned_resumes.append(cleaned)

        jd_emb = runtime.embedder.encode(
            jd_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        resume_embs = runtime.embedder.encode(
            cleaned_resumes,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        ).astype(np.float32)

        features, cosine_sims = build_pair_features_batch(resume_embs, jd_emb)
        probabilities = runtime.model.predict_proba(features)[:, 1]
        predictions = runtime.model.predict(features).astype(int)
        jd_skills = extract_skills(jd_text)

        names = payload.candidate_names or [f"Candidate {i + 1}" for i in range(len(cleaned_resumes))]
        rows: list[RankedCandidate] = []
        for idx, resume_text in enumerate(cleaned_resumes):
            resume_skills = extract_skills(resume_text)
            rows.append(
                RankedCandidate(
                    rank=0,
                    candidate_index=idx,
                    candidate_name=names[idx].strip() or f"Candidate {idx + 1}",
                    match_probability=float(probabilities[idx]),
                    predicted_label=int(predictions[idx]),
                    confidence_band=confidence_band(float(probabilities[idx])),
                    cosine_similarity=float(cosine_sims[idx]),
                    matched_skills=sorted(resume_skills.intersection(jd_skills)),
                    missing_skills=sorted(jd_skills.difference(resume_skills)),
                    resume_text=resume_text,
                )
            )

        ranked = sorted(rows, key=lambda item: item.match_probability, reverse=True)
        for rank_idx, item in enumerate(ranked, start=1):
            item.rank = rank_idx

        return RankBatchResponse(
            jd_text=jd_text,
            total_candidates=len(ranked),
            ranked_candidates=ranked,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Batch ranking failed.")
        raise HTTPException(status_code=500, detail=f"Batch ranking failure: {exc}") from exc
