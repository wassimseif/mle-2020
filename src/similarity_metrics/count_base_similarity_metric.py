# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Union, List

# Project dependencies


# Project imports
import numpy as np

from similarity_metrics.base_similarity_metric import BaseSimilarityMetric


class CountBasedSimilarity(BaseSimilarityMetric):
    def get_similarity(
        self,
        first_features: Union[np.ndarray, List],
        second_features: Union[np.ndarray, List],
    ):
        """
        Matrix multiply 2 input vectors
        Example:
        [3,0,1,0,1]
        [1,0,0,1,0]
        result = [3,0,0,0,0]

        :param first_features: first input vector
        :param second_features: second input vector
        :return: ndarray 
        """
        if not isinstance(first_features, np.ndarray):
            first_features = np.array(first_features)
        if not isinstance(second_features, np.ndarray):
            second_features = np.array(second_features)

        return first_features @ second_features


if __name__ == "__main__":
    CountBasedSimilarity().get_similarity([1, 2, 3], [1, 0, 1])
