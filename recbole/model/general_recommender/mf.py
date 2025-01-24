import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class MF(GeneralRecommender):
    r"""MF is a traditional matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}` is set as model input,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config["RATING_FIELD"]

        # load parameters info
        self.user_embedding_size = config["user_embedding_size"]
        self.item_embedding_size = config["item_embedding_size"]
        self.inter_matrix_type = config["inter_matrix_type"]

        # generate intermediate data
        if self.inter_matrix_type == "01":
            (
                self.history_user_id,
                self.history_user_value,
                _,
            ) = dataset.history_user_matrix()
            (
                self.history_item_id,
                self.history_item_value,
                _,
            ) = dataset.history_item_matrix()
            self.interaction_matrix = dataset.inter_matrix(form="csr").astype(
                np.float32
            )
        elif self.inter_matrix_type == "rating":
            (
                self.history_user_id,
                self.history_user_value,
                _,
            ) = dataset.history_user_matrix(value_field=self.RATING)
            (
                self.history_item_id,
                self.history_item_value,
                _,
            ) = dataset.history_item_matrix(value_field=self.RATING)
            self.interaction_matrix = dataset.inter_matrix(
                form="csr", value_field=self.RATING
            ).astype(np.float32)
        else:
            raise ValueError(
                "The inter_matrix_type must in ['01', 'rating'] but get {}".format(
                    self.inter_matrix_type
                )
            )
        self.max_rating = self.history_user_value.max()

        # define layers
        self.user_embeddings = nn.Embedding(self.n_users, self.user_embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.item_embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.mf_loss = BPRLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        vector = torch.mul(user_emb, item_emb).sum(dim=1)
        return vector

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == "01":
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == "rating":
            label = interaction[self.RATING] * interaction[self.LABEL]
        output = self.forward(user, item)

        label = label / self.max_rating  # normalize the label to calculate BCE loss.
        loss = self.bce_loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        predict = self.sigmoid(self.forward(user, item))
        return predict

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings.weight
        similarity = torch.mm(user_emb, item_emb.t())
        similarity = self.sigmoid(similarity)
        return similarity.view(-1)