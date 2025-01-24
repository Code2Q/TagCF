import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from recbole.model.abstract_recommender import SequentialRecommender

class LRU(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LRU, self).__init__(config, dataset)
        self.embedding_size = config["embedding_size"]
        self.bert_dropout = config["bert_dropout"]
        self.bert_hidden_units = config["bert_hidden_units"]
        self.embedding = LRUEmbedding(self.n_items, self.bert_hidden_units, self.bert_dropout)
        self.loss_fct = nn.CrossEntropyLoss()

        self.model = LRUModel(config, self.n_items)
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, item_seq, item_seq_len, labels=None):
        x, mask = self.embedding(item_seq)
        output = self.model(x, self.embedding.token.weight, mask, labels=labels)
        # 获取最后一个有效位置的索引
        output = self.gather_indexes(output, item_seq_len - 1)

        # [b, seq, hidden]
        return output
        
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.embedding.token.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # test_item = interaction[self.ITEM_ID]
        # test_item_emb, _ = self.embedding(test_item)  # [B H]
        # seq_output = self.forward(item_seq, item_seq_len)

        # scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return 0
    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        scores = torch.matmul(
            seq_output, self.embedding.token.weight.transpose(0, 1)
        )  # [B, item_num]
        return scores




class LRUEmbedding(nn.Module):
    def __init__(self, n_items, bert_hidden_units,bert_dropout):
        super().__init__()
        vocab_size = n_items
        embed_size = bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(bert_dropout)

    def get_mask(self, x):
        return (x > 0)

    def forward(self, x):
        mask = self.get_mask(x)
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask


class LRUModel(nn.Module):
    def __init__(self, config, n_items):
        super().__init__()
        self.hidden_size = config["bert_hidden_units"]
        layers = config["bert_num_blocks"]

        self.lru_blocks = nn.ModuleList([LRUBlock(config) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(n_items))
        
    def forward(self, x, embedding_weight, mask, labels=None):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)
        
        # # prediction layer
        # scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
        return x



class LRUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = config["bert_hidden_units"]
        bert_dropout = config["bert_dropout"]
        bert_attn_dropout = config["bert_attn_dropout"]
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=bert_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4, dropout=bert_dropout)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)