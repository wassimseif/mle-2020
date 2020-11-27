# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

# Project dependencies
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Project imports
from similarity_metrics.base_similarity_metric import BaseSimilarityMetric


class CosineSimilarityMetric(BaseSimilarityMetric):
    def __init__(self):
        super(CosineSimilarityMetric, self).__init__()
        pass

    def get_similarity(self, first_features: np.ndarray, second_features: np.ndarray):
        """
        This will calculate the cosine similarity between 2 vectors/matrices
        :param first_features: first input vector
        :param second_features: second input vector
        :return: the cosine similarity between the inputs. output shape
        depends on the actual input shape
        """
        warnings.warn("This is not a good metric for the basic content base filtering")
        return cosine_similarity(first_features, second_features.T)


if __name__ == "__main__":
    pass
