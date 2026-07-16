"""mappings.data_mappings._occurrence_subject_mapping — subject -> [indices] accumulation."""
import mappings.data_mappings as dm


def test_occurrence_subject_mapping_accumulates_per_subject():
    dm.SUBJECT_TO_OCCURRENCE_MAPPING.clear()
    try:
        dm._occurrence_subject_mapping("s1", 0)
        dm._occurrence_subject_mapping("s1", 5)
        dm._occurrence_subject_mapping("s2", 2)
        assert dm.SUBJECT_TO_OCCURRENCE_MAPPING == {"s1": [0, 5], "s2": [2]}
    finally:
        dm.SUBJECT_TO_OCCURRENCE_MAPPING.clear()
