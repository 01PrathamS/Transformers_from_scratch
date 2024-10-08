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

class MultiHeadAttentionBlock(nn.Module):

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

    @staticmethod 
    def attention(query, key, value, mask, dropout: nn.Dropout): 

        d_k = query.shape[-1] 
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None: 
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) 
        attention_scores = attention_scores.softmax(dim=-1) ## (Batch, h, seq_len, seq_len) 
        attention_scores = dropout(attention_scores) 
        x = attention_scores @ value 
        return x, attention_scores

    def forward(self, q, k, v, mask): 
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model) 
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Split the d_model into h heads 
        query = query.view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), -1, self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.size(0), -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k)   --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)    
        return self.w_o(x) 