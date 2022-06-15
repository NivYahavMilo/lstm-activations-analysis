import os

import pandas as pd
import torch

import config
from supporting_functions import _load_pkl
from enums import Mode, Network
from config import idx_to_clip


class TableBuilder:

    @staticmethod
    def __get_clip_name(clip_index) -> str:
        reduced_index = clip_index
        while idx_to_clip.get(reduced_index) is None:
            reduced_index -= 18
        return idx_to_clip.get(reduced_index)

    @classmethod
    def _attach_activations_per_subject(cls, occurrences: list, activations: torch.tensor) -> pd.DataFrame:
        xp_list = []
        # Create df per subject
        for clip in occurrences:
            x = activations[clip]
            # convert tensor to data-frame
            xp = pd.DataFrame(x.numpy())
            # remove padded rows
            xp = xp[xp.values.sum(axis=1) != 0]
            # assign label
            xp['y'] = cls.__get_clip_name(clip)
            # assign tr by clip
            xp['tr'] = [*xp.index]
            xp_list.append(xp)

        subject_df = pd.concat([df for df in xp_list])
        subject_df.reset_index(drop=True, inplace=True)
        return subject_df

    @staticmethod
    def __create_subject_dir(subject: str, mode: Mode, network: str = ''):
        subject_dir = os.path.join(config.ACTIVATION_MATRICES, subject)
        # Open new directory as subjects id
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        # Open new directory according to activations mode rest/clip
        subject_mode_dir = os.path.join(subject_dir, mode.value)
        if not os.path.exists(subject_mode_dir):
            os.makedirs(subject_mode_dir)

        subject_net_dir = os.path.join(subject_mode_dir, network)
        if network != '':
            if not os.path.exists(subject_net_dir):
                os.makedirs(subject_net_dir)
            return subject_net_dir

        return subject_mode_dir

    @classmethod
    def subject_tables_forming_wb(cls, activation_path, mode: Mode):
        subjects_mappings = _load_pkl(os.path.join(config.MAPPINGS_PATH,
                                                   "subject_occurrence_mapping.pkl"))
        activations = _load_pkl(activation_path)
        activations = activations.get("lstm_activations")

        for subject, occurrence in subjects_mappings.items():
            subjects_dir = cls.__create_subject_dir(str(subject), mode)
            subject_df = cls._attach_activations_per_subject(occurrence, activations)

            # save subject df
            subject_df.to_csv(os.path.join(subjects_dir, 'activation_matrix.csv'), index=False)
            print(f"Done forming table - subject: {subject}; mode: {mode.name}")

    @classmethod
    def subject_tables_forming_networks(cls, activation_path, mode: Mode):
        subjects_mappings = _load_pkl(os.path.join(config.MAPPINGS_PATH,
                                                   "subject_occurrence_mapping.pkl"))
        test_subjects = subjects_mappings.get("test")
        for net in os.listdir(activation_path):
            activations = _load_pkl(os.path.join(activation_path, net))
            activations = activations.get("lstm_activations")
            net = net.replace(' activations.pkl', '')
            for subject, occurrence in test_subjects.items():
                subjects_dir = cls.__create_subject_dir(str(subject), mode, net)
                subject_df = cls._attach_activations_per_subject(occurrence, activations)

                # save subject df
                subject_df.to_csv(os.path.join(subjects_dir, 'activation_matrix.csv'), index=False)
                print(f"Done forming networks table - subject: {subject}; mode: {mode.name}; net: {net}")


if __name__ == '__main__':
    table_builder = TableBuilder()

    path = os.path.join(config.ACTIVATIONS_NETWORKS_PATH, Mode.CLIPS.value)
    table_builder.subject_tables_forming_networks(path, Mode.CLIPS)

    path = os.path.join(config.ACTIVATIONS_NETWORKS_PATH, Mode.REST_BETWEEN.value)
    table_builder.subject_tables_forming_networks(path, Mode.REST_BETWEEN)
