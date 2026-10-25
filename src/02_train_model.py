"""Model training pipeline for Resume-JD semantic matching.

Pipeline stages:
1) Load cleaned data.
2) Generate Sentence-BERT embeddings.
3) Build engineered pairwise features.
4) Tune HistGradientBoostingClassifier using GridSearchCV.
5) Save model artifacts for online inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


LOGGER = logging.getLogger("train_model")


@dataclass(frozen=True)
class TrainConfig:
    """Serializable training configuration persisted with artifacts."""

    sentence_model_name: str = "all-MiniLM-L6-v2"
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 3
    n_jobs: int = -1
    scoring: str = "f1"


def configure_logging() -> None:
    """Configure structured logs for the training job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned training data from CSV or Parquet."""
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet.")

    required_cols = {"resume_text", "jd_text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into dense semantic vectors."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_pair_features(resume_emb: np.ndarray, jd_emb: np.ndarray) -> np.ndarray:
    """Create rich pairwise features from two embedding matrices."""
    if resume_emb.shape != jd_emb.shape:
        raise ValueError(
            f"Embedding shape mismatch. resume={resume_emb.shape}, jd={jd_emb.shape}"
        )

    dot_product = np.sum(resume_emb * jd_emb, axis=1, keepdims=True)
    cosine_sim = dot_product  # vectors are normalized
    l2_distance = np.linalg.norm(resume_emb - jd_emb, axis=1, keepdims=True)
    abs_diff = np.abs(resume_emb - jd_emb)
    hadamard = resume_emb * jd_emb

    features = np.hstack(
        [
            resume_emb,
            jd_emb,
            abs_diff,
            hadamard,
            cosine_sim,
            l2_distance,
        ]
    )
    return features.astype(np.float32)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TrainConfig,
    quick_mode: bool = False,
) -> GridSearchCV:
    """Run hyperparameter tuning with cross-validation."""
    base_model = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=cfg.random_state,
        class_weight="balanced",
    )
    if quick_mode:
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [None, 8],
            "max_leaf_nodes": [31, 63],
            "min_samples_leaf": [20],
            "l2_regularization": [0.0, 0.01],
            "max_iter": [200],
        }
    else:
        param_grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [None, 8, 12],
            "max_leaf_nodes": [31, 63, 127],
            "min_samples_leaf": [20, 40],
            "l2_regularization": [0.0, 0.01, 0.1],
            "max_iter": [200, 300],
        }
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=cfg.scoring,
        cv=cfg.cv_folds,
        n_jobs=cfg.n_jobs,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model: HistGradientBoostingClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate the fitted model on holdout data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    LOGGER.info("Evaluation metrics: %s", metrics)
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return metrics


def persist_artifacts(
    artifacts_dir: Path,
    model: HistGradientBoostingClassifier,
    metrics: dict[str, float],
    cfg: TrainConfig,
) -> None:
    """Save model and metadata for backend inference."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "hist_gbc_model.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    joblib.dump(model, model_path)

    payload = {
        "config": asdict(cfg),
        "metrics": metrics,
        "model_file": model_path.name,
        "feature_spec": [
            "resume_embedding",
            "jd_embedding",
            "absolute_difference",
            "hadamard_product",
            "cosine_similarity",
            "l2_distance",
        ],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Saved model artifact: %s", model_path)
    LOGGER.info("Saved metadata: %s", metadata_path)


def parse_args() -> argparse.Namespace:
    """Parse script CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Resume-JD matching classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/resume_jd_clean.csv"),
        help="Path to processed dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store serialized model artifacts.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced hyperparameter search for faster local validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run complete model training and artifact persistence."""
    configure_logging()
    args = parse_args()
    cfg = TrainConfig()

    df = load_training_data(args.data_path)
    y = df["label"].astype(int).to_numpy()

    sentence_model = SentenceTransformer(cfg.sentence_model_name)
    resume_embeddings = encode_texts(sentence_model, df["resume_text"].astype(str).tolist())
    jd_embeddings = encode_texts(sentence_model, df["jd_text"].astype(str).tolist())
    X = build_pair_features(resume_embeddings, jd_embeddings)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    LOGGER.info("Training samples: %d | Test samples: %d", len(y_train), len(y_test))
    search = train_classifier(X_train, y_train, cfg, quick_mode=args.quick)
    LOGGER.info("Best params: %s", search.best_params_)
    LOGGER.info("Best CV F1: %.4f", search.best_score_)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)
    metrics["cv_best_f1"] = float(search.best_score_)

    persist_artifacts(
        artifacts_dir=args.artifacts_dir,
        model=best_model,
        metrics=metrics,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
