# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from typing import Union

# Project dependencies
import numpy as np
import torch
from scipy.spatial import distance

# Project imports
from similarity_metrics.base_similarity_metric import BaseSimilarityMetric


class CosineSimilarityMetric(BaseSimilarityMetric):
    def __init__(self):
        super(CosineSimilarityMetric, self).__init__()
        pass

    def get_similarity(
        self,
        first_features: Union[np.ndarray, torch.Tensor],
        second_features: Union[np.ndarray, torch.Tensor],
    ):
        """
        This will calculate the cosine similarity between 2 vectors/matrices
        :param first_features: first input vector
        :param second_features: second input vector
        :return: the cosine similarity between the inputs. output shape
        depends on the actual input shape
        """
        warnings.warn("This is not a good metric for the basic content base filtering")

        return 1 - distance.cosine(first_features, second_features)


if __name__ == "__main__":
    pass
