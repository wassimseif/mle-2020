# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

# Project dependencies
import pandas as pd
import numpy as np

# Project imports
from project import Project
from similarity_metrics import (
    BaseSimilarityMetric,
    CosineSimilarityMetric,
    CountBasedSimilarity,
)
from exceptions import DataIntegrityException


class SimpleContentBaseFilteringRecommendations:
    def __init__(
        self,
        products_dataframe: pd.DataFrame,
        ratings_dataframe: pd.DataFrame,
        similarity_metric: BaseSimilarityMetric,
        products_feature_cols: List[str],
    ):
        """
        Returns recommendations based on the content only.

        :param products_dataframe: The dataframe representing the products
        :param ratings_dataframe: The dataframe representing user interactions with the products
        :param similarity_metric: The similarity metric to use when finding similar metrics. Check `similarity_metrics` module for details
        :param products_feature_cols: The features to use when performing this recommendation.
        """
        self.products_data = products_dataframe
        self.ratings_data = ratings_dataframe
        self.similarity_metric = similarity_metric
        self.feature_cols = products_feature_cols
        self.logger = self._setup_logging()
        self.similarity_matrix: Optional[np.ndarray] = None
        self._check_data_integrity()
        self._build_similarity_matrix()

    def _check_data_integrity(self):
        """
        Checks if the features to used are actually available in the products data
        """
        if not pd.Series(self.feature_cols).isin(self.products_data.columns).all():
            raise DataIntegrityException(
                "Some feature columns do no exist in the product data"
            )

    def _build_similarity_matrix(self) -> None:
        """
        Builds the similarity matrix using the `similarity_metric` passed
        """
        self.logger.info("Building similarity matrix")
        self.similarity_matrix = self.similarity_metric.get_similarity(
            self.products_data[self.feature_cols].values,
            self.products_data[self.feature_cols].values.T,
        )

        self.logger.info("Done building similarity matrix")

    def _setup_logging(self) -> logging.Logger:
        """
        Creates and sets up the logger to log errors. This logger will print to
        stdout and logs to a file in the log/ dir with name of file as now()
        formatted Returns: the created logger

        """
        log_file_name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        logger = logging.getLogger("content_base_filtering_logger")
        logger.addHandler(logging.StreamHandler())

        file_handler = logging.FileHandler(f"{Project.log_dir}/{log_file_name}.log")

        file_handler.formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        return logger

    def print_data_shapes(self) -> None:

        print(
            f"Products data has {self.products_data.shape[0]} data points with {self.products_data.shape[1]} features"
        )

    def get_product_name(self, index: int) -> str:
        """
        Returns the name of the product given its index in the similarity matrix
        :param index: index of the requested product in similarity matrix
        :return: str representing the name of the product
        """
        return self.products_data.iloc[index].title

    def get_product_index(self, product_name: str) -> int:
        """
        Given the name of the product, it will return its index in the matrix
        :param product_name: the name of the product to be used
        :return: the index
        """
        res = self.products_data[self.products_data["title"] == product_name]

        if len(res) > 1:

            raise DataIntegrityException(
                "Multiple products with same name found. Recommendation will not be "
                "accurate "
            )
        elif len(res) == 0:
            raise DataIntegrityException(f"No products found for name {product_name}")
        else:
            return res.index[0]

    def get_recommendations(
        self, product_name: str, topk: int = 10, export_csv: bool = False
    ) -> List[Tuple[int, str, float]]:
        """
        Given a product name, this will show the topk similar products. The resulting
        CSV will be saved to exported_csv dir
        :param product_name: Product name to get recommendation for
        :param topk: number of recommendations to get by descending similarity score
        :param export_csv: If the dataframe resulting should be exported or not. default = False 
        :return: List of Tuples representing the product ID ,product name and the recommendation score
        """
        product_id = self.get_product_index(product_name)
        if self.similarity_matrix is None:
            self.logger.info(
                "Trying to get similarity without similarity matrix. building it now "
            )
            self._build_similarity_matrix()
        best = self.similarity_matrix[product_id].argsort()[::-1]
        reco = [
            (ind, self.get_product_name(ind), self.similarity_matrix[product_id, ind],)
            for ind in best[:topk]
            if ind != product_id
        ]

        if export_csv:
            df = pd.DataFrame(reco, columns=["product_id", "product_name", "score"])
            df.to_csv(Project.exported_csv_dir / f"{product_name}.csv")
            self.logger.info(
                f"Exported recommendation csv to "
                f'{Project.exported_csv_dir / f"{product_name}.csv"}'
            )
        return reco

    def get_user_recommendations(
        self, user_id: int, topk: int = 10, export_csv: bool = False
    ) -> List[Tuple[int, str, float]]:
        """
        Given a user id. the function will check the most rated products by this user from the `ratings_data` dataframe
        Then from these products. it will find the most similar products them and recommends them.The resulting
        CSV will be saved to exported_csv dir
        :param user_id:  the user id you want to get recommendation for
        :param topk: number of products to recommend
        :param export_csv: If the dataframe resulting should be exported or not. default = False
        :return: List of Tuples representing the product ID ,product name and the recommendation score
        """
        top_rated_movies = (
            self.ratings_data[ratings_data["user_id"] == user_id]
            .sort_values(by="rating", ascending=False)
            .head(3)["product_id"]
        )
        index = ["product_id", "product_name", "similarity"]
        most_similar = []
        for top_movie in top_rated_movies:
            most_similar += self.get_recommendations(self.get_product_name(top_movie))
        if export_csv:
            df = (
                pd.DataFrame(most_similar, columns=index)
                .drop_duplicates()
                .sort_values(by="similarity", ascending=False)
                .head(topk)
            )
            df.to_csv(Project.exported_csv_dir / f"{user_id}_recommendations.csv")
            self.logger.info(
                f"Exported recommendation csv to "
                f'{Project.exported_csv_dir / f"{user_id}_recommendations.csv"}'
            )
        return most_similar


if __name__ == "__main__":
    products_feature_cols: List[str] = [
        "Animation",
        "Children's",
        "Comedy",
        "Adventure",
        "Fantasy",
        "Romance",
        "Drama",
        "Action",
        "Crime",
        "Thriller",
        "Horror",
        "Sci-Fi",
        "Documentary",
        "War",
        "Musical",
        "Mystery",
        "Film-Noir",
        "Western",
    ]
    users_data = pd.read_csv(Project.data_dir / "users.csv")
    products_data = pd.read_csv(Project.data_dir / "movies.csv")
    # So that I don't hardcode anything related to movies in the class
    products_data.rename(
        {"movie_id": "product_id"}, inplace=True, errors="raise", axis=1
    )
    ratings_data = pd.read_csv(Project.data_dir / "ratings.csv")
    ratings_data.rename(
        {"movie_id": "product_id"}, inplace=True, errors="raise", axis=1
    )
    similarity = CountBasedSimilarity()
    recommender = SimpleContentBaseFilteringRecommendations(
        products_dataframe=products_data,
        ratings_dataframe=ratings_data,
        similarity_metric=similarity,
        products_feature_cols=products_feature_cols,
    )
    recommender.print_data_shapes()
    recommender.get_recommendations("Toy Story", export_csv=True)
    print(*recommender.get_user_recommendations(user_id=0, export_csv=True), sep="\n")
