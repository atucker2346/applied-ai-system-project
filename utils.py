import logging
import re
from io import BytesIO

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Lazy-load the embedding model so imports and tests stay fast."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_text_from_pdf(file) -> str:
    """
    Read text from an uploaded PDF (file-like with read()).
    Raises ValueError on empty extraction or read errors.
    """
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        raw = file.read() if hasattr(file, "read") else file
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("Expected a binary PDF upload.")
        buffer = BytesIO(raw)
        text = ""
        with pdfplumber.open(buffer) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = text.strip()
        if not text:
            raise ValueError("No text could be extracted from the PDF.")
        logger.info("Extracted %d characters from PDF", len(text))
        return text
    except Exception as e:
        logger.exception("PDF extraction failed: %s", e)
        raise ValueError("Could not read this PDF. Try a text-based PDF.") from e


def split_text(text: str) -> list[str]:
    """Split into rough clauses/sentences for retrieval-style matching."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[.!?]+\s*|\n+", str(text))
    return [p.strip() for p in parts if p.strip()]


def get_similarity(text1: str, text2: str) -> float:
    if not (text1 and text2 and text1.strip() and text2.strip()):
        logger.warning("get_similarity called with empty text")
        return 0.0
    model = get_model()
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    score = float(cosine_similarity([emb1], [emb2])[0][0])
    return score


def analyze_resume(resume_text: str, job_text: str) -> list[dict]:
    """
    Retrieval-augmented matching: for each job clause, retrieve the best-matching
    resume clause and attach a confidence score (cosine similarity).
    """
    resume_sentences = split_text(resume_text)
    job_sentences = split_text(job_text)

    if not resume_sentences or not job_sentences:
        logger.warning("analyze_resume: missing sentences (resume=%d job=%d)",
                       len(resume_sentences), len(job_sentences))
        return []

    model = get_model()
    resume_embeddings = model.encode(resume_sentences)
    job_embeddings = model.encode(job_sentences)

    results = []
    for i, job_emb in enumerate(job_embeddings):
        similarities = cosine_similarity([job_emb], resume_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        conf = float(similarities[best_idx])
        results.append({
            "job_requirement": job_sentences[i],
            "best_match": resume_sentences[best_idx],
            "score": conf,
            "confidence": conf,
        })
    logger.info("Analyzed %d job clauses against resume", len(results))
    return results


def quality_check(results: list[dict], min_confidence: float = 0.35) -> dict:
    """
    Post-retrieval validation: aggregate confidence and flag weak matches.
    """
    if not results:
        return {
            "passed": False,
            "reason": "no_clauses",
            "mean_confidence": 0.0,
            "low_confidence_count": 0,
        }
    scores = [r["score"] for r in results]
    mean_c = float(np.mean(scores))
    low = sum(1 for s in scores if s < min_confidence)
    passed = mean_c >= min_confidence and low <= max(1, len(scores) // 2)
    return {
        "passed": passed,
        "mean_confidence": mean_c,
        "low_confidence_count": low,
        "clause_count": len(results),
    }


def run_screening_pipeline(resume_text: str, job_text: str) -> dict:
    """
    Agentic-style pipeline: validate inputs, embed/retrieve, then self-check quality.
    Returns structured result for logging and UI.
    """
    steps = []
    if not job_text or not str(job_text).strip():
        logger.warning("pipeline: empty job description")
        return {"ok": False, "error": "Job description is empty.", "steps": ["validate_inputs"]}

    if not resume_text or not str(resume_text).strip():
        logger.warning("pipeline: empty resume text")
        return {"ok": False, "error": "Resume text is empty.", "steps": ["validate_inputs"]}

    steps.append("validate_inputs")
    overall = get_similarity(resume_text, job_text)
    steps.append("overall_similarity")

    results = analyze_resume(resume_text, job_text)
    steps.append("retrieve_per_requirement")

    qc = quality_check(results)
    steps.append("quality_check")

    logger.info(
        "pipeline complete: overall=%.3f clauses=%d qc_passed=%s",
        overall,
        len(results),
        qc["passed"],
    )
    return {
        "ok": True,
        "overall_score": overall,
        "results": results,
        "quality": qc,
        "steps": steps,
    }
