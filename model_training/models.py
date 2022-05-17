from typing import Callable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Lstm(nn.Module):

    def __init__(self, input_size, hidden_units, lstm_layers, n_class):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, n_class)

        # Store activations
        self.hidden_activations = {}

    def forward(self, x, x_len, max_length=None) -> torch.Tensor:
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True, total_length=max_length)
        y = self.fc(x)
        return y

    def register_hook(self, layer: str):
        def hook(model, input, output) -> Callable:
            self.hidden_activations['lstm_activations'] = input[0].detach()
            self.hidden_activations['linear_activations'] = output.detach()

            return hook

        # Register hooks
        layer = getattr(self, layer)
        layer.register_forward_hook(self.register_hook)
