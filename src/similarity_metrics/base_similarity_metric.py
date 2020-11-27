# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Project dependencies
import numpy as np


# Project imports
class BaseSimilarityMetric:
    def __init__(self):
        pass

    def get_similarity(self, first_features: np.ndarray, second_features: np.ndarray):
        raise NotImplementedError("Implement Custom Similarity metrics in subclasses")


if __name__ == "__main__":
    pass
