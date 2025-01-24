import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import (
    TransformerEncoder,
    # MultiHeadCrossAttention,
    VanillaAttention,
)
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

class LogicRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LogicRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.feat_hidden_dropout_prob = config["feat_hidden_dropout_prob"]
        self.feat_attn_dropout_prob = config["feat_attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"] if "layer_norm_eps" in config else 1e-12
        self.mode = config["mode"]
        self.tag_col_name = config["tag_col_name"]
        self.infer_mode = config["infer_mode"]
        # size_list = [self.inner_size] + self.hidden_size
        # self.mlp_layers = MLPLayers(size_list, dropout=self.hidden_dropout_prob)

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]
        # self.num_feature_field = len(config["selected_features"])

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.lamda = config["lamda"]
        self.branch_factor = config["branch_factor"]
        self.full_tag_infer = config["full_tag_infer"]
        # self.max_item_tag, self.max_user_tag = 7179, 5594
        self.item_feat = dataset.get_item_feature().to(self.device)
 
        self.ut2it = torch.load('u2i_book.pt').to(self.device)
        self.it2ut = torch.load('i2u_book.pt').to(self.device)
        print(self.ut2it.shape, self.it2ut.shape)
#         if self.dataset.dataset_name == 'amazon_movie':
#             self.ut2it = torch.load('u2i_movie.pt').to(self.device).to_dense()
#             self.it2ut = torch.load('i2u_movie.pt').to(self.device).to_dense()
#         else:
#             self.ut2it = torch.load('ut2it_sp.pt').to(self.device).to_dense()
#             self.it2ut = torch.load('it2ut_sp.pt').to(self.device).to_dense()

        self.max_item_tag, self.max_user_tag = self.it2ut.shape

        

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.item_tag_embeddings = nn.Embedding(
            self.max_item_tag, self.hidden_size, padding_idx=0
        )
        self.user_tag_embeddings = nn.Embedding(
            self.max_user_tag, self.hidden_size, padding_idx=0
        )
        

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.feat_hidden_dropout_prob,
            attn_dropout_prob=self.feat_attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.feat_dropout = nn.Dropout(self.feat_hidden_dropout_prob)
        self.prediction_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.tag_loss_fct = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)
        # self.other_parameter_name = ["feature_embed_layer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, item_tag_id_lists):
        item_emb = self.item_embedding(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)
        if self.mode == 'item_tag':
            it_seq_embedding = self.item_tag_embeddings(item_tag_id_lists)
        elif self.mode == 'user_tag':
            it_seq_embedding = self.user_tag_embeddings(item_tag_id_lists)


        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get item_trm_input
        item_emb = item_emb + position_embedding #[b, max_len, hidden]
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
       
#         print(it_seq_embedding.shape)

        feature_emb, attn_weight = self.feature_att_layer(it_seq_embedding) #[b, max_len, hidden]
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.feat_dropout(feature_emb)


        item_trm_output = self.item_trm_encoder(
            item_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B H]

        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.prediction_layer(output_concat)
        return output  # [B H]

    def tag_infer_train(self, target_tags): # input : [b, max_tag_len]
        if self.mode == 'user_tag':
            #ut2it  #[b, max_ut_len, n_it]
            ut2it2ut = torch.matmul(self.ut2it[target_tags], self.it2ut).sum(dim=1)  #[b, n_ut]
            _, topk_ut_indices = torch.topk(ut2it2ut, self.branch_factor, dim=1)
            # topk_ut_freq /= topk_ut_freq.sum(dim=-1, keepdim=True)
            return topk_ut_indices
        else:
            it2ut = self.it2ut[target_tags] #[b, max_ut_len, n_it]
            it2ut2it = torch.matmul(it2ut, self.ut2it).sum(dim=1)  #[b, n_ut]
            _, topk_it_indices = torch.topk(it2ut2it, self.branch_factor, dim=1)
            # topk_ut_freq /= topk_ut_freq.sum(dim=-1, keepdim=True)
            return topk_it_indices



    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq] #(50, 15)

        output = self.forward(item_seq, item_seq_len, item_tag_id_lists) #[b, h]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        target_tags = self.item_feat[self.tag_col_name][pos_items] #[b, max_tag_length]
        neg_target_tags = self.item_feat[self.tag_col_name][neg_items]

        topk_ut_indices_infer = self.tag_infer_train(target_tags)
        topk_ut_indices_infer_neg = self.tag_infer_train(neg_target_tags)
        # all_target_tags = topk_ut_indices_infer
        # all_target_tags_neg = topk_ut_indices_infer_neg
        all_target_tags = torch.cat((target_tags, topk_ut_indices_infer), dim=1)
#         all_target_tags_neg = neg_target_tags
        all_target_tags_neg = torch.cat((neg_target_tags, topk_ut_indices_infer_neg), dim=1)

        if self.mode == 'item_tag':
            # tag_embds = self.item_tag_embeddings(target_tags)
            all_tag_embds = self.item_tag_embeddings(all_target_tags)
            # neg_tag_embds = self.item_tag_embeddings(neg_target_tags)
            all_neg_tag_embds = self.item_tag_embeddings(all_target_tags_neg)
        elif self.mode == 'user_tag':
            # tag_embds = self.user_tag_embeddings(target_tags)
            all_tag_embds = self.user_tag_embeddings(all_target_tags) #[b, n_tags, h]
            # neg_tag_embds = self.user_tag_embeddings(neg_target_tags)
            all_neg_tag_embds = self.user_tag_embeddings(all_target_tags_neg)
        # weighted_embds = (tag_embds * topk_ut_freq_infer.unsqueeze(-1)).sum(dim=1)
        # weighted_emb = (selected_tag_emb * infer_tags_freq.unsqueeze(-1)).sum(dim=1)
        test_item_emb = self.item_embedding.weight
        
        id_logits = torch.matmul(output, test_item_emb.transpose(0, 1))
        id_loss = self.loss_fct(id_logits, pos_items)
        
        tag_logits = torch.mul(output.unsqueeze(1), all_tag_embds).sum(dim=-1) #[b, topk_tags]
        binary_target_tags = torch.ones(tag_logits.shape, device=self.device)

        # # Negative tag logits
        neg_tag_logits = torch.mul(output.unsqueeze(1), all_neg_tag_embds).sum(dim=-1)  # [batch_size, num_negatives, max_tag_len]
        combined_tag_logits = torch.cat((tag_logits, neg_tag_logits), dim=1)
        binary_neg_target_tags = torch.zeros(neg_tag_logits.shape, device=self.device)  # Negative tags are 0
        combined_binary_target_tags = torch.cat((binary_target_tags, binary_neg_target_tags), dim=1)
        tag_loss = self.tag_loss_fct(combined_tag_logits, combined_binary_target_tags)

        item_tag_logits = torch.mul(self.item_embedding(pos_items).unsqueeze(1), all_tag_embds).sum(dim=-1)
        neg_item_tag_logits = torch.mul(self.item_embedding(pos_items).unsqueeze(1), all_neg_tag_embds).sum(dim=-1) 
        # neg_item_tag_logits2 = torch.mul(self.item_embedding(pos_items).unsqueeze(1), neg_tag_embds).sum(dim=-1) 
        

        combined_item_tag_logits = torch.cat((item_tag_logits, neg_item_tag_logits), dim=1)
        pos_target = torch.ones(item_tag_logits.shape, device=self.device)
        neg_target = torch.zeros(neg_item_tag_logits.shape, device=self.device)
        combined_item_tag_target = torch.cat((pos_target, neg_target), dim=1)

        # combined_item_tag_target = torch.zeros(combined_item_tag_logits.shape, device=self.device)
        # combined_item_tag_target[:, :item_tag_logits.shape[1]] = 1
        item_tag_loss = self.tag_loss_fct(combined_item_tag_logits, combined_item_tag_target)
            
        # tag_loss = self.tag_loss_fct(tag_logits, binary_target_tags)
        loss = (1-self.lamda) * id_loss + self.lamda * (tag_loss + item_tag_loss)
        # loss = (1-self.lamda) * id_loss + self.lamda * (tag_loss)

        # tag_scores = torch.mul(tag_output.unsqueeze(1), test_item_tag_emb).sum(dim=1)
        print(f"id loss {id_loss}")
        print(f"tag loss {tag_loss}")
        print(f"item tag loss {item_tag_loss}")
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
      
        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
            # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)

        output = self.forward(item_seq, item_seq_len, item_tag_id_lists) #[b, emb]
        test_item_emb = self.item_embedding(test_item)

        target_tags = self.item_feat[self.tag_col_name][test_item] #[b, max_tags_len]
        if self.mode == 'item_tag':
            # test_item_tag_emb = self.item_tag_embeddings(target_tags) #[b, max_taglen, emb]
            all_tag_emb = self.item_tag_embeddings(target_tags)
            # all_tag_emb = self.item_tag_embeddings.weight
        else:
            # test_item_tag_emb = self.user_tag_embeddings(target_tags)
            all_tag_emb = self.user_tag_embeddings(target_tags)
            # all_tag_emb = self.user_tag_embeddings.weight

        # all_tag_scores = torch.matmul(output, all_tag_emb.transpose(0, 1)) #[b, n_tags]  [4096, 7179]
        # print(all_tag_scores.shape)
        # infer_tags_freq, infer_tags_indices = self.infer_tags(all_tag_scores) # [b, topk_tags]

        # selected_tag_emb = all_tag_emb[infer_tags_indices] #[b, topk_tags, hidden]
        # inner_product = torch.matmul(selected_tag_emb, test_items_emb.t())  # [b, topk_tags, n_items]
        # infer_tags_freq_expanded = infer_tags_freq.unsqueeze(-1)  # [b, topk_tags, 1]
        # infer_average_score = (inner_product * infer_tags_freq_expanded).sum(dim=1) # [b, n_items]
        id_scores = torch.mul(output, test_item_emb).sum(dim=1)  # [B]
        tag_scores = torch.mul(output.unsqueeze(1), all_tag_emb).sum(dim=-1)  #[b, max_tags_len]
        average_tag_scores = tag_scores.mean(dim=1)
        print(f"id scores {id_scores[:10]}, tag scores {average_tag_scores[:10]}")
        return id_scores + average_tag_scores 
        # tag_scores = torch.mul(output.unsqueeze(1), all_tag_emb).sum(dim=-1)  #[b, max_tags_len]
        
        
    def infer_tags(self, all_tag_scores):
        if self.mode == 'user_tag':
            
            topk_ut_values, topk_ut_indices = torch.topk(all_tag_scores, self.branch_factor, dim=1) #[b, topk_user_tags]
            topk_ut_values /= (topk_ut_values.sum(dim=1, keepdim=True) + 1e-8)

            selected_ut2it = self.ut2it[topk_ut_indices] #[b, topk_user_tags, n_items]
            topk_infer_user_tags_expanded = topk_ut_values.unsqueeze(-1) #[b, topk_user_tags, 1]
            aggregated_item_tags = torch.bmm(selected_ut2it.transpose(1, 2), topk_infer_user_tags_expanded).squeeze(-1)  # [b, n_item_tags]
            topk_infer_item_tags, topk_it_indices = torch.topk(aggregated_item_tags, self.branch_factor, dim=-1)  # [b, topk_item_tags]
            topk_infer_item_tags /= (topk_infer_item_tags.sum(dim=1, keepdim=True) + 1e-8)

            selected_it2ut = self.it2ut[topk_it_indices]  # [b, topk_item_tags, n_user_tags]
            topk_infer_item_tags_expanded = topk_infer_item_tags.unsqueeze(-1)  # [b, topk_item_tags, 1]
            aggregated_user_tags = torch.bmm(selected_it2ut.transpose(1, 2), topk_infer_item_tags_expanded).squeeze(-1)  # [b, n_user_tags]
            # topk_freq_infer, topk_indices_infer = torch.topk(frequencies, self.branch_factor, dim=1)
            topk_freq_infer, topk_indices_infer = torch.topk(aggregated_user_tags, self.branch_factor, dim=1)
            topk_freq_infer /= (topk_freq_infer.sum(dim=1, keepdim=True) + 1e-8)
            # topk_indices_infer = topk_ut_indices[topk_indices_infer]

            return topk_freq_infer, topk_indices_infer
        else:
            topk_it_values, topk_it_indices = torch.topk(all_tag_scores, self.branch_factor, dim=1) #[b, topk_user_tags]
            topk_it_values /= (topk_it_values.sum(dim=1, keepdim=True) + 1e-8)

            selected_it2ut = self.it2ut[topk_it_indices] #[b, topk_user_tags, n_items]
            topk_infer_item_tags_expanded = topk_it_values.unsqueeze(-1) #[b, topk_user_tags, 1]
            aggregated_user_tags = torch.bmm(selected_it2ut.transpose(1, 2), topk_infer_item_tags_expanded).squeeze(-1)  # [b, n_item_tags]
            topk_infer_user_tags, topk_ut_indices = torch.topk(aggregated_user_tags, self.branch_factor, dim=-1)  # [b, topk_item_tags]
            topk_infer_user_tags /= (topk_infer_user_tags.sum(dim=1, keepdim=True) + 1e-8)

            selected_ut2it = self.ut2it[topk_ut_indices]  # [b, topk_item_tags, n_user_tags]
            topk_infer_user_tags_expanded = topk_infer_user_tags.unsqueeze(-1)  # [b, topk_item_tags, 1]
            aggregated_item_tags = torch.bmm(selected_ut2it.transpose(1, 2), topk_infer_user_tags_expanded).squeeze(-1)  # [b, n_user_tags]
            topk_freq_infer, topk_indices_infer = torch.topk(aggregated_item_tags, self.branch_factor, dim=1)
            topk_freq_infer /= (topk_freq_infer.sum(dim=1, keepdim=True) + 1e-8)

            return topk_freq_infer, topk_indices_infer



    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]



        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
        output = self.forward(item_seq, item_seq_len, item_tag_id_lists)
        test_items_emb = self.item_embedding.weight #[n_items, hidden]
        id_scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B, n_items]

        target_tags = self.item_feat[self.tag_col_name] #[b, true_tags] [10396, 19]
        if self.mode == 'item_tag':
            # test_item_tag_emb = self.item_tag_embeddings(target_tags) #[b, max_taglen, emb]
            all_tag_emb = self.item_tag_embeddings.weight
        else:
            # test_item_tag_emb = self.user_tag_embeddings(target_tags)
            all_tag_emb = self.user_tag_embeddings.weight
        all_tag_scores = torch.matmul(output, all_tag_emb.transpose(0, 1)) #[b, n_tags]  [4096, 7179]
        all_tag_scores_sig = self.sigmoid(all_tag_scores)
        
        if self.full_tag_infer:
            if self.mode == 'user_tag':
                infer_item_tags = torch.matmul(all_tag_scores_sig, self.ut2it) #[b, n_it]
                infer_item_tags /= (infer_item_tags.sum(dim=1, keepdim=True) + 1e-8)
                infer_tags = torch.matmul(infer_item_tags, self.it2ut)
                infer_tags /= (infer_tags.sum(dim=1, keepdim=True) + 1e-8)
            else:
                infer_user_tags = torch.matmul(all_tag_scores_sig, self.it2ut) #[b, n_ut]
                infer_user_tags /= (infer_user_tags.sum(dim=1, keepdim=True) + 1e-8)
                infer_tags = torch.matmul(infer_user_tags, self.ut2it)
                infer_tags /= (infer_tags.sum(dim=1, keepdim=True) + 1e-8)

            # weighted_emb = (all_tag_emb.unsqueeze(0) * infer_tags.unsqueeze(-1)).sum(dim=1) #[b, hidden]
            # # selected_tag_emb = all_tag_emb.unsqueeze(0).expand(infer_user_tags.shape[0], -1, -1) # [b, n_ut, hidden]
            weighted_emb = (all_tag_emb.unsqueeze(0) * infer_tags.unsqueeze(-1) ).sum(dim=1) # [b, hidden]
            infer_average_score = torch.matmul(weighted_emb, test_items_emb.t()) # [b, n_items]
            # weighted_scores = (tag_item_scores * infer_tags_freq.unsqueeze(-1)).sum(dim=1)

            # infer_average_score = (infer_average_score * infer_tags.unsqueeze(-1)).sum(dim=1) # [b, n_items]
        else:
            infer_tags_freq, infer_tags_indices = self.infer_tags(all_tag_scores_sig) # [b, topk_tags]
#             selected_tag_emb = all_tag_emb[infer_tags_indices]  # [b, topk_tags, hidden]
#             tag_item_scores = torch.matmul(selected_tag_emb, test_items_emb.t())  # [b, topk_tags, n_items]
#             infer_average_score = (tag_item_scores * infer_tags_freq.unsqueeze(-1)).sum(dim=1)  # [b, n_items]
            # infer_average_score = tag_item_scores.mean(dim=1)
            selected_tag_emb = all_tag_emb[infer_tags_indices] #[b, topk_tags, hidden]
            weighted_emb = (selected_tag_emb * infer_tags_freq.unsqueeze(-1)).sum(dim=1)  # [b, hidden]
            infer_average_score = torch.matmul(weighted_emb, test_items_emb.t())  # [b, n_items]
            # infer_tags_freq_expanded = infer_tags_freq.unsqueeze(-1)  # [b, topk_tags, 1]
            print(f"infer tag freq")
            print(torch.max(infer_tags_freq), torch.min(infer_tags_freq), torch.mean(infer_tags_freq))
            # infer_average_score = (inner_product * infer_tags_freq_expanded).sum(dim=1) # [b, n_items]
        # infer_average_score = inner_product.mean(dim=1)
       
        all_tag_scores = all_tag_scores.unsqueeze(1).expand(-1, self.n_items, -1) # batch_size, n_items, n_tags]
        selected_scores = torch.gather(all_tag_scores, 2, target_tags.unsqueeze(0).expand(id_scores.shape[0], -1, -1))
        average_tag_scores = selected_scores.mean(dim=2)

        print(f"id scores {torch.max(id_scores)}, {torch.min(id_scores)}, {id_scores[:10]}")  # [b, n_items]
        print(f"tag scores {torch.max(average_tag_scores)}, {torch.min(average_tag_scores)}, {average_tag_scores[:10]}")  # [b, n_items]
        print(f"infer tag scores {torch.max(infer_average_score)}, {torch.min(infer_average_score)}, {infer_average_score[:10]}")  # [b, n_items]
        return id_scores + average_tag_scores  + 0.01 * infer_average_score
        # return id_scores 