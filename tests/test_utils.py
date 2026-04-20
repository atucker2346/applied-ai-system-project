import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import utils


def test_split_text_empty():
    assert utils.split_text("") == []
    assert utils.split_text("   ") == []


def test_split_text_splits_on_period_and_newline():
    t = "First line. Second line!\nThird part?"
    parts = utils.split_text(t)
    assert "First line" in parts
    assert "Second line" in parts
    assert "Third part" in parts


def test_get_similarity_empty_returns_zero():
    assert utils.get_similarity("", "hello") == 0.0
    assert utils.get_similarity("hello", "") == 0.0


def test_analyze_resume_empty_job():
    assert utils.analyze_resume("Some resume. Two.", "") == []


def test_analyze_resume_empty_resume():
    assert utils.analyze_resume("", "Need Python. Need AWS.") == []


@patch.object(utils, "get_model")
def test_get_similarity_with_mock_model(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    mock_get_model.return_value = mock_model
    utils._model = None  # reset if any
    s = utils.get_similarity("a", "b")
    assert s == pytest.approx(1.0, abs=1e-5)


@patch.object(utils, "get_model")
def test_analyze_resume_retrieval_scores(mock_get_model):
    mock_model = MagicMock()
    # Two resume sentences, two job sentences -> encode called with combined list twice?
    # analyze_resume calls encode(resume_sentences) once and encode(job_sentences) once
    mock_model.encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # resume
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # job - same as resume for predictable argmax
    ]
    mock_get_model.return_value = mock_model
    utils._model = None
    out = utils.analyze_resume("A. B.", "A. B.")
    assert len(out) == 2
    assert all("confidence" in r and "best_match" in r for r in out)


def test_quality_check_empty():
    qc = utils.quality_check([])
    assert qc["passed"] is False
    assert qc["reason"] == "no_clauses"


def test_quality_check_with_results():
    results = [{"score": 0.8}, {"score": 0.7}]
    qc = utils.quality_check(results)
    assert qc["mean_confidence"] == pytest.approx(0.75)
    assert qc["passed"] is True


def test_run_screening_pipeline_validation():
    bad = utils.run_screening_pipeline("", "")
    assert bad["ok"] is False
    bad2 = utils.run_screening_pipeline("resume text", "  \n  ")
    assert bad2["ok"] is False


@patch.object(utils, "get_model")
def test_run_screening_pipeline_happy_path(mock_get_model):
    mock_model = MagicMock()
    vec = np.array([1.0, 0.0])
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    # get_similarity: two full-document encodings, then analyze_resume: two batch encodings
    mock_model.encode.side_effect = [vec, vec, emb, emb]
    mock_get_model.return_value = mock_model
    utils._model = None
    out = utils.run_screening_pipeline("Line one. Line two.", "Req one. Req two.")
    assert out["ok"] is True
    assert "overall_score" in out
    assert len(out["results"]) == 2
    assert "quality_check" in out["steps"]
