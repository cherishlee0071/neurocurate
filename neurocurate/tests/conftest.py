import numpy as np
import pandas as pd
import pytest
from neurocurate.eeg import EEGTrials


@pytest.fixture
def trials():
    # Two repetitions per image; baseline=1, poststimulus values differ.
    data = np.array([[[1, 1, 3, 5], [1, 1, 5, 7]],
                     [[1, 1, 5, 7], [1, 1, 7, 9]],
                     [[1, 1, 7, 9], [1, 1, 9, 11]],
                     [[1, 1, 9, 11], [1, 1, 11, 13]]], dtype=float)
    return EEGTrials(data, np.array(["b", "b", "a", "a"]),
                     np.array([-0.2, -0.1, 0, 0.1]), np.array(["O1", "O2"]), "sanity")


@pytest.fixture
def manifest(tmp_path):
    return pd.DataFrame({"image_id": ["a", "b"], "image_path": [str(tmp_path / "a.png"), str(tmp_path / "b.png")],
                         "split": ["sanity", "sanity"], "eeg_index": [0, 1]})
