# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime

# Project dependencies
import transformers
import pandas as pd
import torch
from tqdm import tqdm
from exceptions import DataIntegrityException
from similarity_metrics import CosineSimilarityMetric


# Project imports
from project import Project
from bert_embedding_generator import BertEmbeddingGenerator


class BertContentBaseFiltering:
    embedding_map: Dict[int, torch.Tensor] = {}

    def __init__(
        self,
        bert_model_name: str,
        products_data: pd.DataFrame,
        feature_columns: List[str],
        lazy_load_model: bool = False,
    ):
        """
        This class will use BERT model to generate recommendation based on similarities between products
        How it works:
            From a product, it generates a sentence describing it ( since we don't have product description now
            Passes this sentence through a vanilla bert model
            Takes the last 2 hidden layers and average thems.
            Used the output as an embedding vector
            When asked for a recommendation on a product. It loads/generates all the embeddings for all products
            does cosine similarity between these embeddings and saved the top10 in a CSV file
        :param bert_model_name: the name of the bert model to be used
        :param products_data: products dataframe
        :param feature_columns: The features to use when performing this recommendation.
        :param lazy_load_model: controls when the model is loaded. useful for cold starting systems
        """
        self.model: Optional[transformers.BertModel] = None
        self.tokenizer: Optional[transformers.BertTokenizer] = None
        self.products_data = products_data
        self.bert_model_name = bert_model_name
        self.feature_columns = feature_columns
        self.similarity_metric = CosineSimilarityMetric()
        self.logger = self._setup_logging()

        if not lazy_load_model:
            self._load_model(self.bert_model_name)
        self._generate_embeddings_for_all_products()

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

    def _generate_embeddings_for_all_products(self):
        """
        Uses `BertEmbeddingGenerator` to get the embeddings for all products. saves the results in the `embedding_map` hash map
        """
        if self.model is None:
            self._load_model(self.bert_model_name)
        self.logger.info("Generating BERT embeddings for all products")
        for index, product in tqdm(
            products_dataframe.iterrows(), total=len(products_dataframe)
        ):
            em = BertEmbeddingGenerator().generate_embedding(
                self.model,
                tokenizer=self.tokenizer,
                product=product,
                feature_columns=self.feature_columns,
            )
            product_id = int(product["product_id"])
            self.embedding_map[product_id] = em
        self.logger.info("Done generating BERT embeddings for all products")

    def _load_model(self, bert_model_name: str):
        """
        Loads BERT model and tokenizer
        :param bert_model_name:  the name of the bert model to be used
        """
        self.logger.info(f"Loading model {bert_model_name}")
        self.model = transformers.BertModel.from_pretrained(
            bert_model_name, output_hidden_states=True
        )
        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_name)

    def get_recommendation(self, product_name: str):
        """
            Give a product name. It get the embedding and does cosine similarity with all other embeddings
            saves a CSV file wit the top 10 results
        :param product_name: Name of the product of find the similarity for
        """
        product_id = self.get_product_index(product_name)
        prod_embd = self.embedding_map[product_id]
        index = ["product_id", "product_name", "similarity"]
        output_data: List[Tuple[int, str, float]] = []
        for k, v in self.embedding_map.items():
            if k == product_id:
                continue
            sim = self.similarity_metric.get_similarity(prod_embd, v)
            output_data.append((k, self.get_product_name(k), sim))
        df = (
            pd.DataFrame(output_data, columns=index)
            .drop_duplicates()
            .sort_values(by="similarity", ascending=False)
            .head(10)
        )
        df.to_csv(Project.exported_csv_dir / f"{product_id}_bert_recommendations.csv")
        self.logger.info(
            f"Exported recommendation csv to "
            f'{Project.exported_csv_dir / f"{product_id}_bert_recommendations.csv"}'
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
    products_dataframe = pd.read_csv(Project.data_dir / "movies.csv")
    products_dataframe.rename(
        {"movie_id": "product_id"}, inplace=True, errors="raise", axis=1
    )
    bert_reco = BertContentBaseFiltering(
        bert_model_name="bert-base-uncased",
        products_data=products_dataframe,
        feature_columns=products_feature_cols,
    )
    bert_reco.get_recommendation("Toy Story")
