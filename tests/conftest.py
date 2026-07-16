"""Shared test setup for the LSTM-activations-analysis suite.

Tests exercise the pure arithmetic / algorithmic functions on small synthetic inputs. Some
production code calls ``matplotlib.pyplot.show()``, so a non-interactive backend keeps the run
headless. The repo root is added to ``sys.path`` so top-level modules import as in production.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")  # no interactive windows during tests
