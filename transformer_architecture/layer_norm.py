## batch of items
## for each item in batch 
## calculate mean and variance independently for each item  
## normalize each item independently with its mean and variance
## we are also adding two learning two parameters gamma and beta for each item in the batch
## gamma and beta are learnable parameters

import torch 
import torch.nn as nn 

class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  ## adding this for numerical stability
        self.alpha = nn.Parameter(torch.ones(1)) ## Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) ## Additive 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) 
        std = x.std(dim=-1, keepdim=True) 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
if __name__ == "__main__":
    ln = LayerNormalization()
    x = torch.randn(2, 3, 4)
    print(ln(x).shape) # torch.Size([2, 3, 4])


