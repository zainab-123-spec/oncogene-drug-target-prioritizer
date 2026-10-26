"""QC rules for IC50 rows: units, censoring, and deduplication."""

from __future__ import annotations

import pytest

from odtp.qc import (
    apply_qc,
    deduplicate_pairs,
    pic50_from_nm,
    sample_offsets,
    to_nanomolar,
)
from tests.conftest import measurement


def test_pic50_ten_nanomolar_is_8() -> None:
    assert pic50_from_nm(10) == pytest.approx(8.0)


def test_micromolar_converts_to_nanomolar_and_pic50() -> None:
    nanomolar = to_nanomolar(1, "uM")
    assert nanomolar == pytest.approx(1000)
    assert nanomolar is not None
    assert pic50_from_nm(nanomolar) == pytest.approx(6.0)


def test_millimolar_conversion() -> None:
    assert to_nanomolar(1, "mM") == pytest.approx(1_000_000)
    assert pic50_from_nm(1_000_000) == pytest.approx(3.0)


def test_micro_symbol_units() -> None:
    assert to_nanomolar(2, "µM") == pytest.approx(2000)


def test_rejects_censored_relation() -> None:
    frame = apply_qc([measurement(standard_relation=">", activity_id=1)])
    assert frame.empty


def test_rejects_unknown_units() -> None:
    frame = apply_qc([measurement(standard_units="ug.mL-1")])
    assert frame.empty


def test_rejects_nonpositive_ic50() -> None:
    frame = apply_qc([measurement(standard_value=0)])
    assert frame.empty
    with pytest.raises(ValueError):
        pic50_from_nm(-5)


def test_rejects_validity_comment() -> None:
    frame = apply_qc([measurement(data_validity_comment="Outside typical range")])
    assert frame.empty


def test_rejects_potential_duplicate() -> None:
    frame = apply_qc([measurement(potential_duplicate=1)])
    assert frame.empty


def test_exact_nanomolar_row_is_kept() -> None:
    frame = apply_qc([measurement()])
    assert len(frame) == 1
    assert frame.loc[0, "pIC50"] == pytest.approx(8.0)
    assert frame.loc[0, "active"] == 1


def test_dedup_uses_median_pic50() -> None:
    rows = apply_qc(
        [
            measurement(activity_id=1, standard_value=10, molecule_chembl_id="CHEMBL1"),
            measurement(activity_id=2, standard_value=100, molecule_chembl_id="CHEMBL2"),
        ]
    )
    deduped = deduplicate_pairs(rows)
    assert len(deduped) == 1
    assert deduped.loc[0, "pIC50"] == pytest.approx(7.5)
    assert deduped.loc[0, "n_measurements"] == 2


def test_dedup_collapses_parent_ids() -> None:
    rows = apply_qc(
        [
            measurement(
                parent_molecule_chembl_id="PARENT",
                molecule_chembl_id="SALT1",
                activity_id=1,
            ),
            measurement(
                parent_molecule_chembl_id="PARENT",
                molecule_chembl_id="SALT2",
                activity_id=2,
                standard_value=100,
            ),
        ]
    )
    deduped = deduplicate_pairs(rows)
    assert len(deduped) == 1
    assert deduped.loc[0, "parent_molecule_chembl_id"] == "PARENT"


def test_active_threshold_uses_median() -> None:
    rows = apply_qc(
        [
            measurement(activity_id=1, standard_value=500),
            measurement(activity_id=2, standard_value=2000, molecule_chembl_id="CHEMBL9"),
        ]
    )
    deduped = deduplicate_pairs(rows)
    # 500 nM -> 6.30, 2000 nM -> 5.70, median 6.0, which is active.
    assert deduped.loc[0, "pIC50"] == pytest.approx(6.0)
    assert deduped.loc[0, "active"] == 1


def test_inactive_below_threshold() -> None:
    frame = apply_qc([measurement(standard_value=2000)])
    assert frame.loc[0, "pIC50"] == pytest.approx(pic50_from_nm(2000))
    assert frame.loc[0, "active"] == 0


def test_sample_offsets_paginates_small_results() -> None:
    assert sample_offsets(500, 200, 3) == [0, 200, 400]


def test_sample_offsets_spaces_large_archives() -> None:
    offsets = sample_offsets(24925, 200, 3)
    assert offsets[0] == 0
    assert offsets[-1] == 24925 - 200
    assert len(offsets) == 3
