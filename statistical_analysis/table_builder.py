import os

import pandas as pd
import torch

import config
from supporting_functions import _load_pkl
from enums import Mode
from config import idx_to_clip


class TableBuilder:

    def __init__(self, mode: Mode = Mode.CLIPS):
        self.table = None
        self.subjects_mappings_test: dict = _load_pkl(
            os.path.join(config.MAPPINGS_PATH, 'subject_occurrence_mapping.pkl')).get('test')
        self.activations: torch.Tensor = _load_pkl(
            os.path.join(config.MODELS_PATH, f'{mode.value}_model_activations.pkl')).get('lstm_activations')
        self.mode = mode.value

    def _get_clip_name(self, clip_index) -> str:
        reduced_index = clip_index
        while idx_to_clip.get(reduced_index) is None:
            reduced_index -= 18
        return idx_to_clip.get(reduced_index)

    def _attach_activations_per_subject(self, occurrences: list) -> pd.DataFrame:
        xp_list = []
        # Create df per subject
        for clip in occurrences:
            x = self.activations[clip]
            # convert tensor to data-frame
            xp = pd.DataFrame(x.numpy())
            # remove padded rows
            xp = xp[xp.values.sum(axis=1) != 0]
            # assign label
            xp['y'] = self._get_clip_name(clip)
            # assign tr by clip
            xp['tr'] = [*xp.index]
            xp_list.append(xp)

        subject_df = pd.concat([df for df in xp_list])
        subject_df.reset_index(drop=True, inplace=True)
        return subject_df

    def _create_subject_dir(self, subject: str):
        subject_dir = os.path.join(config.ACTIVATION_MATRICES, subject)
        # Open new directory as subjects id
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        # Open new directory according to activations mode rest/clip
        subject_mode_dir = os.path.join(subject_dir, self.mode)
        if not os.path.exists(subject_mode_dir):
            os.makedirs(subject_mode_dir)

        return subject_mode_dir

    @classmethod
    def subject_tables_forming_wb(self):
        for subject,occurrence in self.subjects_mappings_test.items():
            subjects_dir = self._create_subject_dir(str(subject))
            subject_df = self._attach_activations_per_subject(occurrence)

            # save subject df
            subject_df.to_csv(os.path.join(subjects_dir, 'activation_matrix.csv'), index=False)

    @classmethod
    def subject_tables_forming_networks(self):
        for subject,occurrence in self.subjects_mappings_test.items():
            subjects_dir = self._create_subject_dir(str(subject))
            subject_df = self._attach_activations_per_subject(occurrence)

            # save subject df
            subject_df.to_csv(os.path.join(subjects_dir, 'activation_matrix.csv'), index=False)


if __name__ == '__main__':
    table_builder = TableBuilder(Mode.CLIPS)
    table_builder.subject_tables_forming()
    table_builder = TableBuilder(Mode.REST_BETWEEN)
    table_builder.subject_tables_forming()



