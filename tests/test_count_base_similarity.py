# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Project dependencies
import numpy as np
import pytest

# Project imports
from src.similarity_metrics import CountBasedSimilarity


def test_count_base_similarity():
    sim = CountBasedSimilarity()
    first_features = np.array([1, 0, 0, 0, 1, 1.0])
    second_features = np.array([1, 0, 0, 0, 1, 1.0])
    assert sim.get_similarity(first_features, second_features) == 3

    first_features = np.array([0, 0, 0, 0, 0, 0])
    second_features = np.array([1, 0, 0, 0, 1, 1.0])
    assert sim.get_similarity(first_features, second_features) == 0

