# Python standard lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Optional

# Project dependencies
import torch
import transformers
import pandas as pd

# Project imports
from project import Project


class BertEmbeddingGenerator:
    def __init__(self):
        pass

    def generate_product_description(
        self, product: pd.Series, feature_columns: List[str]
    ) -> str:
        """
        THIS IS JUST A PROOF OF CONCEPT. This function will generate a description of a product from the set of available features.
        Normally you have a product description to use directly
        :param product: product to generate the description for
        :param feature_columns: The feature columns to use when generating this description
        :return: str representing the description
        """
        desc = self.get_hard_coded_description(product)

        for col in feature_columns:
            if int(product[col]) == 1:
                desc += f" {col} "
        return desc

    def load_already_geneated_embedding(
        self, product: pd.Series
    ) -> Optional[torch.Tensor]:
        """
        Embedding for this product is already generated. load it directly
        :param product: product to get the embedding for
        :return: torch.Tensor for the embedding
        """
        tt = torch.load(Project.exported_objects_dir / f"{product['product_id']}.obj")

        return tt

    def generate_embedding(
        self,
        model: transformers.BertModel,
        tokenizer: transformers.BertTokenizer,
        product: pd.Series,
        feature_columns: List[str],
    ) -> torch.Tensor:
        model.eval()
        if (Project.exported_objects_dir / f"{product['product_id']}.obj").exists():
            return self.load_already_geneated_embedding(product=product)
        product_description = self.generate_product_description(
            product=product, feature_columns=feature_columns
        )
        marked_text = "[CLS] " + product_description + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        torch.save(
            sentence_embedding,
            Project.exported_objects_dir / f"{product['product_id']}.obj",
        )
        return sentence_embedding

    def get_hard_coded_description(self, product: pd.Series) -> str:
        """
        Could not figure out a hack for this in the time being.
        :param product:
        :return: Title of the product and when it was produced
        """
        tmp = ""
        tmp += str(product["title"])
        tmp += " produced in " + str(product["year"])
        return tmp
