import logging
import sys

import streamlit as st

from utils import extract_text_from_pdf, run_screening_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("AI Resume Screener")

st.sidebar.markdown(
    "### Human review\n"
    "Scores are **semantic similarity**, not hiring decisions. "
    "Use this tool to triage candidates; final judgment should involve people."
)

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=200)

if st.button("Run screening", type="primary"):
    if not resume_file:
        st.warning("Upload a PDF resume first.")
    elif not job_description or not job_description.strip():
        st.warning("Paste a job description.")
    else:
        try:
            resume_text = extract_text_from_pdf(resume_file)
        except ValueError as e:
            logger.info("Resume upload rejected: %s", e)
            st.error(str(e))
        else:
            try:
                out = run_screening_pipeline(resume_text, job_description)
            except Exception as e:
                logger.exception("Screening failed: %s", e)
                st.error("Something went wrong during analysis. Check logs for details.")
            else:
                if not out["ok"]:
                    st.error(out.get("error", "Unknown error"))
                else:
                    overall = out["overall_score"]
                    pct = round(overall * 100, 2)
                    st.subheader(f"Overall match score: {pct}%")
                    st.caption(
                        "Interpretation: embedding similarity between full resume and full job text "
                        "(higher suggests more topical overlap)."
                    )

                    qc = out["quality"]
                    if qc["passed"]:
                        st.success(
                            f"Quality check: mean confidence {qc['mean_confidence']:.2f} "
                            f"across {qc['clause_count']} job clauses."
                        )
                    else:
                        st.warning(
                            f"Quality check: several weak matches (mean confidence "
                            f"{qc['mean_confidence']:.2f}; {qc['low_confidence_count']} "
                            f"below threshold). Treat results as tentative."
                        )

                    st.subheader("Detailed analysis (retrieval-augmented)")
                    st.caption(
                        "For each job clause we retrieve the closest resume clause and show "
                        "**confidence** (same as cosine similarity)."
                    )

                    for r in out["results"]:
                        conf = r["confidence"]
                        label = f"{r['job_requirement'][:120]}{'…' if len(r['job_requirement']) > 120 else ''}"
                        detail = (
                            f"**Best resume evidence:** {r['best_match'][:300]}"
                            f"{'…' if len(r['best_match']) > 300 else ''}\n\n"
                            f"**Confidence:** {conf:.2f}"
                        )
                        if conf > 0.6:
                            st.success(f"**Matched:** {label}\n\n{detail}")
                        elif conf > 0.35:
                            st.warning(f"**Partial:** {label}\n\n{detail}")
                        else:
                            st.error(f"**Weak / missing:** {label}\n\n{detail}")

                    with st.expander("Pipeline trace (agentic steps)"):
                        st.write(out.get("steps", []))
