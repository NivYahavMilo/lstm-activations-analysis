from supporting_functions import _dict_to_pkl

SUBJECT_TO_OCCURRENCE_MAPPING = {}

def _occurrence_subject_mapping(subject: str, idx: int):
    global SUBJECT_TO_OCCURRENCE_MAPPING
    SUBJECT_TO_OCCURRENCE_MAPPING.setdefault(subject,[]).append(idx)

