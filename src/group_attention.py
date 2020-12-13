import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
# from .transformer_part import group_mask,src_to_mask


# 按逗号分割，相同的子句中设置为同一个token，遇到新逗号时token+1, 问题句子的token设置为1000
def src_to_mask(src, comma_index, full_stop_index):
    # src shape: [batch_size, seq_len]
    src = src.cpu().numpy()
    batch_data_mask_tok = []
    for encode_sen_idx in src:

        token = 1
        mask = [0] * len(encode_sen_idx)
        for num in range(len(encode_sen_idx)):
            mask[num] = token
            if (encode_sen_idx[num] == comma_index or encode_sen_idx[num] == full_stop_index) \
                    and num != len(encode_sen_idx) - 1:
                token += 1
            if encode_sen_idx[num] == 0:
                mask[num] = 0
        for num in range(len(encode_sen_idx)):
            if mask[num] == token and token != 1:
                mask[num] = 1000
        batch_data_mask_tok.append(mask)
    # [[1, 1, 1, 2, 2, 2, 3, 3,1000, 1000, 0, 0,...], [1, 1, 1, 2, 2, 2, 3, 3,1000, 1000, 0, 0,...],  ... ]
    return np.array(batch_data_mask_tok)


# batch: 已mask为1，2，3, (1000),0,0,0....的batch
def group_mask(batch, mask_type="self", pad=0):
    length = batch.shape[1]
    lis = []
    # 当前子句的设置为1， 其他设置为0
    if mask_type == "self":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask,-1)
            # print('mask:', mask, mask.shape, tok.shape)
            for ele in tok:
                if ele == pad:
                    copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:
                        copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                    # print("self copy",copy)
                '''
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                '''
                copy = np.expand_dims(copy,-1)
                mask = np.concatenate((mask,copy),axis=1)  # 将copy按列叠加到Mask中
            mask = mask[:, 1:]  # 第一列是全0
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)  # mask shape: [1, seq_len, seq_len]
            lis.append(mask)
        res = np.concatenate(tuple(lis))  # res shape: [batch_size, seq_len, seq_len]
    # 当前子句和问题句子设置为0，其他子句设置为1
    elif mask_type == "between":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask, -1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy == 1000] = 0
                    copy[copy == ele] = 0
                    copy[copy != 0] = 1
                    '''
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    '''
                copy = np.expand_dims(copy, -1)
                mask = np.concatenate((mask, copy), axis=1)
            mask = mask[:, 1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif mask_type == "question":
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask, -1)
            for ele in tok:
                if ele == pad:copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele == 1000:  # question部分设置为0, 其他部分设置为1
                    copy[copy == 0] = -1
                    copy[copy == 1] = 0
                    copy[copy == -1] = 1
                copy = np.expand_dims(copy, -1)
                mask = np.concatenate((mask, copy), axis=1)
            mask = mask[:, 1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:return "error"
    return res


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    add & norm
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             /math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class GroupAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, src, comma_index, full_stop_index, pad=0):
        mask = src_to_mask(src, comma_index, full_stop_index)
        src_mask_self = torch.from_numpy(group_mask(mask, "self", pad).astype('uint8')).unsqueeze(1)
        src_mask_between = torch.from_numpy(group_mask(mask, "between", pad).astype('uint8')).unsqueeze(1)
        src_mask_question = torch.from_numpy(group_mask(mask, "question", pad).astype('uint8')).unsqueeze(1)
        src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        src_mask_global = src_mask_global.expand(src_mask_self.shape)
        # print('src shape:',  src.size())
        # print('src_mask_self shape:', self.src_mask_self.size())
        # print('src_mask_between shape:', self.src_mask_between.size())
        # print('src_mask_question shape:', self.src_mask_question.size())
        # print('src_mask_global shape:', self.src_mask_question.size())
        final = torch.cat((src_mask_between.cuda(), src_mask_self.cuda(),
                                src_mask_global.cuda(), src_mask_question.cuda()), 1)

        # print('final shape', self.final.size())
        return final.cuda()

    def forward(self, query, key, value, mask=None):
        #print("query",query,"\nkey",key,"\nvalue",value)
        "Implements Figure 2"

        if mask is not None and len(mask.shape)<4:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # which is linears(query, key, value)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
