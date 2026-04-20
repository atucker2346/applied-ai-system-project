# AI Resume Screener

# Link to DEMO - https://www.loom.com/share/d94fcb9c756d44f095bbe518b8a78f9e

## Original project (Modules 1–3)

**Replace this with your real repo name and URL**, e.g. `github.com/yourusername/module-1-3-resume-helper`.

The earlier project focused on basic resume handling and exploratory NLP: extracting text from documents and experimenting with simple matching ideas. This repository **evolves** that work into a small, end-to-end **applied AI system** with retrieval-style analysis, a structured screening pipeline, automated tests, logging, and documentation suitable for a portfolio.

---

## Title and summary

**AI Resume Screener** compares an uploaded resume (PDF) to a pasted job description using **semantic embeddings**. It reports an **overall similarity score** and, for each job clause, **retrieves** the closest matching sentence from the resume with an explicit **confidence** (cosine similarity). A **quality check** flags when many matches are weak so users do not over-trust the scores.

This matters because hiring teams need fast triage tools, but those tools must be **transparent** (show evidence from the resume), **guardrailed** (errors and empty PDFs handled), and **testable** so behavior does not silently regress.

---

## Architecture overview

Main components and data flow:

1. **Human** uploads a PDF and pastes a job description in **Streamlit**.
2. **PDF extraction** (`pdfplumber`) turns the file into plain text (with validation).
3. **Screening pipeline** validates inputs, computes overall similarity, runs **per-clause retrieval**, then runs a **quality check**.
4. **Embeddings** (`SentenceTransformer` `all-MiniLM-L6-v2`) encode resume and job text.
5. **Retrieval-augmented analysis**: for each job clause, the system picks the best resume clause and attaches **confidence**; the UI shows that evidence, not only a single number.
6. **Humans and tests**: sidebar reminds reviewers that scores are not decisions; **pytest** verifies core logic with mocked embeddings.

![System architecture](assets/architecture.svg)

---

## Setup instructions

**Prerequisites:** Python **3.10+** (tested on 3.12), `pip`, and ~500MB+ disk for the first model download.

```bash
cd applied-ai-system-project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the app:**

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

**Run tests:**

```bash
python3 -m pytest tests/ -v
```

The first app run downloads the embedding model from the network; ensure outbound HTTPS is allowed.

---

## Sample interactions

**Example 1 — strong overlap**

- **Input (job, excerpt):** “We need strong Python experience. The role requires AWS and Docker for deployment.”
- **Resume (excerpt):** “Senior engineer with 5 years Python. Built services on AWS ECS and packaged apps with Docker.”
- **Typical output:** High overall match; per-clause rows show **Matched** with high **confidence** and resume sentences about Python / AWS / Docker as **best resume evidence**.

**Example 2 — partial overlap**

- **Input (job):** “Must have Kubernetes and Go.”
- **Resume:** “Python and Django developer; some Linux administration.”
- **Typical output:** Lower overall score; Kubernetes/Go clauses show **Weak / missing** with low confidence; quality check may warn that many matches are weak.

**Example 3 — bad PDF**

- **Input:** Scanned image-only PDF with no text layer.
- **Typical output:** Error message: no text extracted; logs record the failure (no silent failure).

---

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **SentenceTransformer** (general-purpose MiniLM) | Fast, runs locally, no API key | Not fine-tuned to “hiring” language; domain jargon can be mis-ranked |
| **Clause splitting** with regex | Keeps dependencies small | Imperfect segmentation vs. a heavy NLP sentence splitter |
| **Cosine similarity as confidence** | Interpretable, standard for embeddings | Not calibrated probability; “0.7” ≠ “70% chance correct” |
| **Pipeline + quality check** | Surfaces uncertainty after retrieval | Thresholds (`quality_check`) are heuristic and should be tuned with real data |
| **pytest with mocked `encode`** | CI-friendly, deterministic | Does not integration-test the real model (only that wiring works) |

---

## Testing summary

Automated: **11 pytest** cases cover text splitting, empty inputs, mocked embedding similarity, retrieval shape, quality aggregation, and pipeline validation. Latest local run: **11 passed**.

Qualitative: the system **struggles** when the job description is one long bullet-less paragraph (weak clause split) or when the resume uses rare abbreviations the model does not align with the JD. **Confidence scores** average higher when both texts use similar vocabulary; adding stricter validation (minimum clause length, language detection) would be the next step.

One-line summary for reviewers: *11/11 unit tests passed; embedding confidence is useful for ranking but should not replace human review when context is thin or PDF text is noisy.*

---

## Reflection

Building this reinforced that **retrieval-style** UX (show the **matched resume line**) builds more trust than a single opaque score. Logging made debugging PDF and shape issues much faster.

**Limitations and bias:** Pretrained embeddings encode **corpus biases** (e.g., certain roles or demographics associated with wording). The model can favor fluent, keyword-rich resumes over equally qualified but terse ones. Non-English text is not explicitly handled.

**Misuse and mitigation:** The tool could be misused to **auto-reject** candidates from a single threshold. Mitigations: prominent **human-in-the-loop** copy, confidence + evidence display, quality warnings, and no auto-decision API in this demo.

**What surprised me:** Small formatting changes in the JD (line breaks vs. one block) changed clause splits and therefore per-clause scores more than expected—**chunking strategy** matters as much as the model.

---

## Collaboration with AI (course prompt)

- **Helpful:** AI suggested structuring the screening flow as explicit **steps** (validate → score → retrieve → check), which cleaned up the Streamlit code and made logging easier to reason about.
- **Flawed:** An early suggestion used aggressive sentence splitting that broke short job requirements into empty fragments; **human validation** plus unit tests on edge cases corrected that.

---

## Project layout

```
applied-ai-system-project/
  app.py              # Streamlit UI
  utils.py            # PDF, embeddings, retrieval, pipeline, quality check
  requirements.txt
  pytest.ini
  assets/
    architecture.svg  # System diagram
  tests/
    test_utils.py
```

---

## License

Add a license if you publish publicly (this repo did not ship with one by default).
