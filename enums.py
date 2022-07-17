from enum import Enum


class Mode(Enum):
    CLIPS = 'clips'
    # REST = 'rest'
    REST_BETWEEN = 'rest_between'
    COMBINED = 'combined'


class Network(Enum):
    WB = 'whole brain'
    Visual = 'VisualNetwork'
    Limbic = 'Limbic'
    Somatomotor = 'SomMotor'
    DorsalAttention = 'DorsalAttention'
    VentralAttention = 'SalVenAttn'
    Default = 'DMN'
    Frontoparietal = 'Cont'


class DataType(Enum):
    FMRI = 'fmri'
    LSTM_PATTERNS = 'lstm patterns'
