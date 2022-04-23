import os
import config
from supporting_functions import _load_pkl
import pandas as pd
import torch

def attach_activations_to_its_subject(activation:dict, layer: str, subjects_list: list):
    acts: torch.Tensor = activation.get(f"{layer}_activations")
    num_of_occurrences = acts.shape[0]
    occurrences_per_subject = num_of_occurrences // config.TEST_SUBJECTS_AMOUNT  # 14 clips, 4 testretest

    start=0
    for sub in range(0, num_of_occurrences, occurrences_per_subject):
        occurrence_length = []
        single_sub = acts[sub:sub+occurrences_per_subject, :, :]
        for i in range(0, occurrences_per_subject):
            occurrence = single_sub[i]
            for tr in range(occurrence.shape[0]):
                if all((occurrence[tr] == torch.zeros(150)).tolist()) or tr == occurrence.shape[0] - 1:
                    occurrence_length.append(tr)
                    break
        print(occurrence_length)
        print(len(occurrence_length))


if __name__ == '__main__':
    activations_: dict = _load_pkl(os.path.join(config.MODELS_PATH, 'clip_model_activations.pkl'))
    subject_lst = pd.read_pickle(os.path.join(config.DATA_PATH, '4_movie_runs.pkl'))['Subject'].unique()[100:].tolist()
    attach_activations_to_its_subject(activations_, 'lstm', subject_lst)
