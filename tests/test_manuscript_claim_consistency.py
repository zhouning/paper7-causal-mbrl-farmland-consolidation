import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
LEDGER_PATH = (
    REPO_ROOT
    / "paper7"
    / "results"
    / "full_rigor"
    / "manuscript_evidence_ledger.json"
)
MANUSCRIPT_PATHS = [
    REPO_ROOT / "submission" / "ceus" / "01_main_document_anonymous" / "manuscript.tex",
    REPO_ROOT
    / "submission"
    / "ceus"
    / "06_latex_source_editable"
    / "manuscript_anonymous_copy.tex",
    REPO_ROOT
    / "submission"
    / "ceus"
    / "06_latex_source_editable"
    / "manuscript_signed.tex",
]
HIGHLIGHTS_PATH = REPO_ROOT / "submission" / "ceus" / "03_highlights" / "highlights.txt"
COVER_LETTER_PATH = (
    REPO_ROOT / "submission" / "ceus" / "04_cover_letter" / "cover_letter.txt"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _read_all_manuscripts() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS}


def test_manuscript_uses_audited_paired_calibration_p_values():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    paired = audit["evidence"]["bishan_seed_chain"]["paired_slope_test"]
    one_sided = f"p={paired['one_sided_p']:.3f}"
    two_sided = f"p={paired['two_sided_p']:.3f}"

    for path, manuscript in _read_all_manuscripts().items():
        compact = _compact(manuscript)
        assert "0.004" not in manuscript, path
        assert one_sided in compact, path
        assert two_sided in compact, path
        assert "Mann-Whitney" not in manuscript, path
        assert "Mann--Whitney" not in manuscript, path


def test_manuscript_keeps_review_boundaries_visible():
    required = [
        "observational reward regularization",
        "not definitive causal identification",
        "not direct bishan-to-dongxing policy transfer",
        "descriptive",
        "model-free",
        "real-environment evaluation",
        "fixed-policy",
    ]

    for path, manuscript in _read_all_manuscripts().items():
        lower = manuscript.lower()
        for phrase in required:
            assert phrase in lower, f"{phrase!r} missing from {path}"


def test_manuscript_does_not_reintroduce_forbidden_overclaims():
    forbidden_patterns = [
        r"universal generalization across counties",
        r"formal superiority over all model-free",
        r"direct transfer of bishan policies to dongxing",
        r"definitive causal identification of reward effects",
        r"transition model as a replacement for final real-environment evaluation",
    ]

    for path, manuscript in _read_all_manuscripts().items():
        lower = manuscript.lower()
        for pattern in forbidden_patterns:
            assert re.search(pattern, lower) is None, f"{pattern!r} found in {path}"


def test_generated_ledger_supports_manuscript_claims():
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    claim_ids = {row["claim_id"] for row in ledger["claims"]}

    assert "calibration_effect" in claim_ids
    assert "dongxing_local_counterpart" in claim_ids
    assert "direct_transfer_boundary" in claim_ids
    assert "reward_weight_replay_boundary" in claim_ids
    assert "not definitive causal identification" in ledger["required_boundaries"]
    assert "not direct Bishan-to-Dongxing policy transfer" in ledger[
        "required_boundaries"
    ]


def test_highlights_and_cover_letter_use_bounded_full_rigor_framing():
    highlights = HIGHLIGHTS_PATH.read_text(encoding="utf-8").lower()
    cover = COVER_LETTER_PATH.read_text(encoding="utf-8").lower()

    assert "local counterpart" in highlights
    assert "transfer" not in highlights
    assert "descriptive comparison" in cover
    assert "direct cross-county transfer" in cover
    assert "observational" in cover

