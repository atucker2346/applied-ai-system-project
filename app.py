import streamlit as st
from utils import extract_text_from_pdf, analyze_resume, get_similarity

st.title("AI Resume Screener")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if resume_file and job_description:
    resume_text = extract_text_from_pdf(resume_file)

    overall_score = get_similarity(resume_text, job_description)
    st.subheader(f"Match Score: {round(overall_score * 100, 2)}%")

    results = analyze_resume(resume_text, job_description)

    st.subheader("Detailed Analysis")

    for r in results:
        if r["score"] > 0.6:
            st.success(f"Matched: {r['job_requirement']}")
        else:
            st.error(f"Missing: {r['job_requirement']}")