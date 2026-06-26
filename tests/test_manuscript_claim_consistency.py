import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
MANUSCRIPT_PATH = (
    REPO_ROOT
    / "submission"
    / "ceus"
    / "01_main_document_anonymous"
    / "manuscript.tex"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_manuscript_uses_audited_paired_calibration_p_values():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    compact = _compact(manuscript)

    paired = audit["evidence"]["bishan_seed_chain"]["paired_slope_test"]
    one_sided = f"p={paired['one_sided_p']:.3f}"
    two_sided = f"p={paired['two_sided_p']:.3f}"

    assert "0.004" not in manuscript
    assert one_sided in compact
    assert two_sided in compact
    assert "Mann-Whitney" not in manuscript
    assert "Mann--Whitney" not in manuscript


def test_manuscript_keeps_review_boundaries_visible():
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8").lower()

    assert "observational reward regularization" in manuscript
    assert "not definitive causal identification" in manuscript
    assert "not direct bishan-to-dongxing policy transfer" in manuscript
    assert "descriptive" in manuscript
    assert "model-free" in manuscript
