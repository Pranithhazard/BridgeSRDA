import pandas as pd
import pytest

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.analyze_stride2_cycling import delivered_rows


def fake_rows():
    return pd.DataFrame([{"cycle_index": center, "lead": lead, "target_cycle_index": center + lead}
                         for center in range(4, 800, 8) for lead in range(5) if center + lead < 800])


def test_delivery_has_no_future_updated_analysis_substitution():
    delivered = delivered_rows(fake_rows())
    assert len(delivered) == 199
    assert delivered.lead.value_counts().to_dict() == {0: 100, 4: 99}
    first_skip = delivered.iloc[1]
    assert first_skip.target_cycle_index == 8
    assert first_skip.cycle_index == 4 and first_skip.lead == 4


def test_delivery_rejects_duplicate_or_missing_targets():
    rows = fake_rows()
    with pytest.raises(ValueError):
        delivered_rows(pd.concat([rows, rows.iloc[:1]]))
    with pytest.raises(ValueError):
        delivered_rows(rows.iloc[1:])
