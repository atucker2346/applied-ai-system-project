import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once (important for performance)
model = SentenceTransformer('all-MiniLM-L6-v2')


# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


# -----------------------------
# Split text into sentences
# -----------------------------
def split_text(text):
    return [s.strip() for s in text.split('.') if s.strip()]


# -----------------------------
# Get overall similarity score
# -----------------------------
def get_similarity(text1, text2):
    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return score


# -----------------------------
# Detailed resume analysis
# -----------------------------
def analyze_resume(resume_text, job_text):
    resume_sentences = split_text(resume_text)
    job_sentences = split_text(job_text)

    # Handle edge cases
    if not resume_sentences or not job_sentences:
        return []

    resume_embeddings = model.encode(resume_sentences)
    job_embeddings = model.encode(job_sentences)

    results = []

    for i, job_emb in enumerate(job_embeddings):
        similarities = cosine_similarity([job_emb], resume_embeddings)[0]
        best_idx = np.argmax(similarities)

        results.append({
            "job_requirement": job_sentences[i],
            "best_match": resume_sentences[best_idx],
            "score": float(similarities[best_idx])
        })

    return results