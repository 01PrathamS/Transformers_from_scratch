import torch 
import torch.nn as nn 

from attention import MultiHeadAttentionBlock 
from feed_forward import FeedForwardBlock 
from residual_con import ResidualConnection 
from layer_norm import LayerNormalization 

class EncoderBlock(nn.Module): 

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None: 
        super().__init__() 
        self.self_attention_block = self_attention_block 
        self.feed_forward_block = feed_forward_block 
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask): 
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x 
    

class Encoder(nn.Module): 

    def __init__(self, layers: nn.ModuleList) -> None: 
        super().__init__() 
        self.layers = layers 
        self.norm = LayerNormalization() 

    def forward(self, x, mask): 
        for layer in self.layers: 
            x = layer(x, mask) 
        return self.norm(x)

    
