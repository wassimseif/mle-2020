from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from project import Project
from typing import List
import pandas as pd
from similarity_metrics import CountBasedSimilarity
from simple_content_base_filtering import SimpleContentBaseFilteringRecommendations

"""
These tests are just preliminary. The approach will be simple.
Using some manual labor. We will know that Toy Story and Adventures of Rocky and Bullwinkle are similar.
We will test that our model recommends Adventures of Rocky and Bullwinkle when asked to generate the recommendations for Toy story

"""


def test_toy_story_recommendation():
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
    rec = recommender.get_recommendations("Toy Story", export_csv=False)

    assert rec[0][1] == 'Space Jam'
    assert rec[0][2] == 3.0

    assert rec[1][1] == 'Adventures of Rocky and Bullwinkle, The'
    assert rec[1][2] == 3.0
    print(rec)

