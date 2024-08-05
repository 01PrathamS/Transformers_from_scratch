import torch 
import torch.nn as nn
import math 

def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, scores

## here we are not showing it for batches, we are doing it for one sentence, but in transformers it is done for batches

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None: 
        super().__init__() 
        self.d_model = d_model 
        self.h = h 
        assert d_model % h == 0, "d_model must be divisible by h" 

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) ## Wq
        self.w_k = nn.Linear(d_model, d_model) ## Wk 
        self.w_v = nn.Linear(d_model, d_model) ## Wv
        self.w_o = nn.Linear(d_model, d_model) ## Wo 

        self.dropout = nn.Dropout(dropout)

