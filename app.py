"""Streamlit frontend for Resume-JD matching.

The frontend is intentionally decoupled from ML execution and interacts with the backend
exclusively through REST APIs.
"""

from __future__ import annotations

import html
import os
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PyPDF2 import PdfReader


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/predict"
RANK_BATCH_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/rank_batch"
MAX_PREVIEW_CHARS = 4000
WORD_BOUNDARY_TEMPLATE = r"\b{}\b"


def inject_global_styles() -> None:
    """Apply production-grade visual styling for the app shell and components."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #0f172a 0%, #020617 60%, #020617 100%);
            color: #e2e8f0;
        }
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem;
            color: #f8fafc;
        }
        .subtitle {
            color: #94a3b8;
            margin-bottom: 1.2rem;
            font-size: 0.97rem;
        }
        .glass-panel {
            border: 1px solid #1e293b;
            background: linear-gradient(180deg, #0b1220 0%, #0b1324 100%);
            border-radius: 14px;
            padding: 1rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 8px 30px rgba(2, 6, 23, 0.35);
        }
        .glass-label {
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .glass-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .summary-card {
            border: 1px solid #1e293b;
            background: linear-gradient(180deg, #0b1220 0%, #0a1120 100%);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 35px rgba(2, 6, 23, 0.35);
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 1rem 0 0.65rem 0;
            color: #f8fafc;
        }
        .skill-panel {
            border-radius: 12px;
            border: 1px solid #1e293b;
            padding: 0.8rem 0.9rem 0.95rem 0.9rem;
            min-height: 112px;
        }
        .skill-panel h4 {
            margin: 0 0 0.65rem 0;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .skill-panel-green { background: rgba(22, 163, 74, 0.08); }
        .skill-panel-green h4 { color: #86efac; }
        .skill-panel-red { background: rgba(239, 68, 68, 0.10); }
        .skill-panel-red h4 { color: #fca5a5; }
        .skill-chip {
            display: inline-block;
            margin: 0.15rem 0.3rem 0.15rem 0;
            padding: 0.22rem 0.5rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .skill-chip-green {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .skill-chip-red {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fca5a5;
        }
        .doc-grid-title {
            font-size: 1.28rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.55rem;
            color: #f8fafc;
        }
        .legend {
            color: #cbd5e1;
            margin-bottom: 0.8rem;
            font-size: 0.9rem;
        }
        .legend-pill {
            display: inline-block;
            margin-right: 0.55rem;
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .legend-match {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        .legend-missing {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .doc-header {
            font-size: 1.03rem;
            font-weight: 700;
            margin: 0.2rem 0 0.45rem 0;
            color: #f8fafc;
        }
        .doc-box {
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 14px 15px;
            max-height: 580px;
            overflow-y: auto;
            background: #f8fafc;
            color: #0f172a;
            line-height: 1.58;
            font-size: 0.97rem;
            white-space: pre-wrap;
            box-shadow: 0 6px 24px rgba(2, 6, 23, 0.28);
        }
        .matched {
            background-color: #bbf7d0;
            color: #14532d;
            padding: 1px 4px;
            border-radius: 4px;
            font-weight: 600;
        }
        .missing {
            background-color: #fecaca;
            color: #7f1d1d;
            padding: 1px 4px;
            border-radius: 4px;
            font-weight: 600;
        }
        .stSidebar [data-testid="stFileUploaderDropzone"] {
            background: #0f172a;
            border: 1px solid #334155;
        }
        .stSidebar .stTextArea textarea {
            background: #0b1220;
            color: #e2e8f0;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from uploaded PDF bytes."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Extract text from uploaded TXT or PDF files."""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix == ".pdf":
        return extract_text_from_pdf(raw_bytes)
    if suffix in {".txt", ".md"}:
        return raw_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, TXT, or MD files.")


def build_relevancy_gauge(score: float) -> go.Figure:
    """Build a Plotly gauge chart for relevance score."""
    percentage = max(0.0, min(score * 100.0, 100.0))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 38, "color": "#e2e8f0"}},
            title={"text": "Relevancy Score"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#0b1220",
                "steps": [
                    {"range": [0, 50], "color": "#451a1a"},
                    {"range": [50, 75], "color": "#422006"},
                    {"range": [75, 100], "color": "#052e16"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "value": percentage,
                },
            },
        )
    )
    figure.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
    )
    return figure


def _wrap_keywords(text: str, words: list[str], css_class: str) -> str:
    """Highlight a keyword list in plain text and return escaped HTML."""
    escaped_text = html.escape(text)
    if not words:
        return escaped_text

    unique_words = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    highlighted = escaped_text
    for word in unique_words:
        escaped_word = re.escape(html.escape(word))
        pattern = re.compile(WORD_BOUNDARY_TEMPLATE.format(escaped_word), flags=re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f"<span class='{css_class}'>{m.group(0)}</span>",
            highlighted,
        )
    return highlighted


def _render_skill_chips(skills: list[str], chip_class: str) -> str:
    """Render a list of skills as chips."""
    if not skills:
        return "<span style='color:#94a3b8;'>No skills detected.</span>"
    return "".join(
        f"<span class='skill-chip {chip_class}'>{html.escape(skill)}</span>" for skill in skills
    )


def render_side_by_side(resume_text: str, jd_text: str, matched_skills: list[str], missing_skills: list[str]) -> None:
    """Render side-by-side document views with skill highlights."""
    col_left, col_right = st.columns(2)
    resume_render = _wrap_keywords(resume_text[:MAX_PREVIEW_CHARS], matched_skills, "matched")
    jd_highlighted = _wrap_keywords(jd_text[:MAX_PREVIEW_CHARS], matched_skills, "matched")
    for keyword in sorted({k.strip() for k in missing_skills if k.strip()}, key=len, reverse=True):
        escaped_keyword = re.escape(html.escape(keyword))
        pattern = re.compile(WORD_BOUNDARY_TEMPLATE.format(escaped_keyword), flags=re.IGNORECASE)
        jd_highlighted = pattern.sub(
            lambda m: f"<span class='missing'>{m.group(0)}</span>",
            jd_highlighted,
        )

    with col_left:
        st.markdown("<div class='doc-header'>Resume Review</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='doc-box'>{resume_render}</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown("<div class='doc-header'>JD Review</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='doc-box'>{jd_highlighted}</div>", unsafe_allow_html=True)


def call_predict_api(resume_text: str, jd_text: str) -> dict:
    """Invoke backend prediction endpoint and return parsed JSON."""
    response = requests.post(
        PREDICT_ENDPOINT,
        json={"resume_text": resume_text, "jd_text": jd_text},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.json()


def call_rank_batch_api(jd_text: str, resumes: list[str], candidate_names: list[str]) -> dict:
    """Invoke backend batch ranking endpoint and return parsed JSON."""
    response = requests.post(
        RANK_BATCH_ENDPOINT,
        json={
            "jd_text": jd_text,
            "candidate_resumes": resumes,
            "candidate_names": candidate_names,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"API error {response.status_code}: {response.text}")
    return response.json()


def resolve_jd_text(
    input_mode: str,
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None,
    pasted_text: str,
) -> str | None:
    """Resolve JD text from upload or pasted mode."""
    if input_mode == "Upload File":
        if uploaded_file is None:
            return None
        return extract_text(uploaded_file)
    cleaned = pasted_text.strip()
    return cleaned or None


def resolve_resume_batch(
    input_mode: str,
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile] | None,
    pasted_text: str,
) -> tuple[list[str], list[str]]:
    """Resolve candidate resume texts and labels from upload or pasted mode."""
    if input_mode == "Upload Files":
        if not uploaded_files:
            return [], []
        texts: list[str] = []
        names: list[str] = []
        for file in uploaded_files:
            text = extract_text(file)
            cleaned = text.strip()
            if cleaned:
                texts.append(cleaned)
                names.append(file.name)
        return texts, names

    blocks = [block.strip() for block in re.split(r"\n\s*---+\s*\n", pasted_text) if block.strip()]
    if not blocks:
        return [], []
    names = [f"Pasted Resume {i + 1}" for i in range(len(blocks))]
    return blocks, names


def _render_input_summary(resume_count: int, has_jd_text: bool) -> None:
    """Render a compact input readiness summary."""
    resume_status = f"{resume_count} loaded" if resume_count else "Missing"
    jd_status = "Ready" if has_jd_text else "Missing"
    st.markdown(
        "<div class='glass-panel'>"
        "<div class='glass-label'>Input Status</div>"
        f"<div class='glass-value'>Resumes: {resume_status} | JD: {jd_status}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _build_leaderboard_frame(ranked_candidates: list[dict]) -> pd.DataFrame:
    """Convert ranked candidate payload into display-friendly dataframe."""
    rows = []
    for row in ranked_candidates:
        rows.append(
            {
                "Rank": row["rank"],
                "Candidate": row["candidate_name"],
                "Match %": round(row["match_probability"] * 100, 2),
                "Confidence": row["confidence_band"],
                "Cosine Sim": round(row["cosine_similarity"], 4),
                "Predicted Label": row["predicted_label"],
            }
        )
    return pd.DataFrame(rows)


def _validate_text_lengths(jd_text: str | None, resumes: list[str]) -> str | None:
    """Return validation error if text content is too short, else None."""
    if not jd_text or len(jd_text.strip()) < 20:
        return "Job Description text is too short. Please provide richer content."
    if not resumes:
        return "At least one resume is required."
    for idx, resume in enumerate(resumes):
        if len(resume.strip()) < 20:
            return f"Resume #{idx + 1} is too short. Please upload richer content."
    return None


def main() -> None:
    """Render Streamlit application."""
    st.set_page_config(page_title="AI Resume-JD Matcher", page_icon="🧠", layout="wide")
    inject_global_styles()
    if "ranking_result" not in st.session_state:
        st.session_state["ranking_result"] = None
    if "ranking_jd_text" not in st.session_state:
        st.session_state["ranking_jd_text"] = ""

    st.markdown("<div class='main-title'>AI-Powered Resume ↔ JD Matching</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Batch ranking dashboard for recruiter-grade Resume ↔ JD matching and candidate deep-dives.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Inputs & Ranking")
        st.caption("Upload many resumes and rank them against one JD.")

        st.markdown("#### Job Description Input")
        jd_mode = st.radio("JD Source", ["Upload File", "Paste Text"], key="jd_source")
        jd_file = None
        jd_paste = ""
        if jd_mode == "Upload File":
            jd_file = st.file_uploader(
                "Upload Job Description (PDF/TXT/MD)",
                type=["pdf", "txt", "md"],
                key="jd",
            )
        else:
            jd_paste = st.text_area(
                "Paste JD Text",
                key="jd_paste",
                height=220,
                placeholder="Paste full job description text here...",
            )

        st.markdown("#### Candidate Resumes")
        resume_mode = st.radio("Resume Source", ["Upload Files", "Paste Text"], key="resume_source")
        resume_files: list[st.runtime.uploaded_file_manager.UploadedFile] | None = None
        resume_paste = ""
        if resume_mode == "Upload Files":
            resume_files = st.file_uploader(
                "Upload Resumes (PDF/TXT/MD)",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                key="resume_batch",
            )
        else:
            resume_paste = st.text_area(
                "Paste Resume Text Blocks",
                key="resume_paste",
                height=220,
                placeholder="Paste multiple resumes separated by lines containing only ---",
            )

        run_analysis = st.button("Rank Candidates", type="primary", use_container_width=True)
        clear_analysis = st.button("Clear Ranking", use_container_width=True)
        st.caption(f"Endpoints: {RANK_BATCH_ENDPOINT} | {PREDICT_ENDPOINT}")

    if clear_analysis:
        st.session_state["ranking_result"] = None
        st.session_state["ranking_jd_text"] = ""

    try:
        jd_text = resolve_jd_text(jd_mode, jd_file, jd_paste)
        resume_texts, resume_names = resolve_resume_batch(resume_mode, resume_files, resume_paste)
    except Exception as exc:
        st.error(f"Failed to parse inputs: {exc}")
        return

    _render_input_summary(len(resume_texts), bool(jd_text))

    if run_analysis:
        validation_error = _validate_text_lengths(jd_text, resume_texts)
        if validation_error:
            st.warning(validation_error)
            return

        with st.spinner("Scoring semantic relevance..."):
            try:
                result = call_rank_batch_api(
                    jd_text=jd_text or "",
                    resumes=resume_texts,
                    candidate_names=resume_names,
                )
            except Exception as exc:
                st.error(f"Batch ranking request failed: {exc}")
                return
        st.session_state["ranking_result"] = result
        st.session_state["ranking_jd_text"] = jd_text or ""

    ranking_result = st.session_state.get("ranking_result")
    if not ranking_result:
        st.info("Load one JD and multiple resumes, then click **Rank Candidates**.")
        return

    ranked_candidates = ranking_result["ranked_candidates"]
    st.markdown("<div class='section-title'>Ranked Leaderboard</div>", unsafe_allow_html=True)
    st.dataframe(_build_leaderboard_frame(ranked_candidates), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Candidate Deep-Dive</div>", unsafe_allow_html=True)
    options = [f"#{c['rank']} — {c['candidate_name']}" for c in ranked_candidates]
    selected_label = st.selectbox("Select candidate for detailed review", options=options, index=0)
    selected_rank = int(selected_label.split(" — ")[0].replace("#", "").strip())
    selected = next(row for row in ranked_candidates if row["rank"] == selected_rank)

    top_a, top_b = st.columns([2, 1])
    with top_a:
        st.plotly_chart(build_relevancy_gauge(selected["match_probability"]), use_container_width=True)
    with top_b:
        st.markdown(
            "<div class='glass-panel'>"
            "<div class='glass-label'>Candidate Summary</div>"
            f"<div class='glass-value'>{selected['candidate_name']}</div>"
            f"<div style='margin-top:0.5rem; color:#cbd5e1;'>Rank: <b>#{selected['rank']}</b></div>"
            f"<div style='margin-top:0.2rem; color:#cbd5e1;'>Confidence: <b>{selected['confidence_band']}</b></div>"
            f"<div style='margin-top:0.2rem; color:#cbd5e1;'>Cosine Similarity: <b>{selected['cosine_similarity']:.4f}</b></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Match Probability", f"{selected['match_probability']*100:.2f}%")
    kpi2.metric("Predicted Label", str(selected["predicted_label"]))
    kpi3.metric("Leaderboard Rank", f"#{selected['rank']}")

    st.markdown("<div class='section-title'>Skill Gap Insights</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div class='skill-panel skill-panel-green'><h4>Matched Skills</h4>"
            f"{_render_skill_chips(selected['matched_skills'], 'skill-chip-green')}</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            "<div class='skill-panel skill-panel-red'><h4>Missing Skills</h4>"
            f"{_render_skill_chips(selected['missing_skills'], 'skill-chip-red')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='doc-grid-title'>Side-by-Side Document Review</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='legend'>"
        "<span class='legend-pill legend-match'>Matched</span>"
        "<span class='legend-pill legend-missing'>Missing</span>"
        "Keyword highlights are extracted from the skill-gap engine."
        "</div>",
        unsafe_allow_html=True,
    )
    render_side_by_side(
        resume_text=selected["resume_text"],
        jd_text=st.session_state["ranking_jd_text"],
        matched_skills=selected["matched_skills"],
        missing_skills=selected["missing_skills"],
    )


if __name__ == "__main__":
    main()
