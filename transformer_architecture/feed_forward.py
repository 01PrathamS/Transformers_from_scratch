import torch
import torch.nn as nn

# as per the paper, d_model(embedding_size) = 512, d_ff(hidden_layer) = 2048

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

## it takes input as batch, seq_len, d_model    
if __name__ == "__main__":
    ff = FeedForwardBlock(512, 2048, 0.1)
    x = torch.randn(2, 3, 512)
    print(ff(x).shape) # torch.Size([2, 3, 512])
    
