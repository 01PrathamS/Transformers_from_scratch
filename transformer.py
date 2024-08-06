import torch 
import torch.nn as nn 

from attention import MultiHeadAttentionBlock 
from feed_forward import FeedForwardBlock 
from residual_con import ResidualConnection 
from layer_norm import LayerNormalization 
from embedding import PositionalEncoding, InputEmbedding 
from encoder import EncoderBlock, Encoder 
from decoder import DecoderBlock, Decoder

class ProjectionLayer(nn.Module): 

    def __init__(self, d_model: int, vocab_size: int) -> None: 
        super().__init__() 
        self.proj = nn.Linear(d_model, vocab_size) 

    def forward(self, x): 
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size) 
        return torch.log_softmax(self.proj(x), dim=-1) 
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbedding, 
                 tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
                 
        super().__init__() 
        self.encoder = encoder 
        self.decoder = decoder 
        self.src_embed = src_embed 
        self.tgt_embed = tgt_embed 
        self.src_pos = src_pos 
        self.tgt_pos = tgt_pos 
        self.projection_layer = projection_layer 

    def encoder(self, src, src_mask): 
        src = self.src_embed(src) 
        src = self.src_pos(src)  
        return self.encoder(src, src_mask)   
    
    def decoder(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt) 
        tgt = self.tgt_pos(tgt) 
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    
    
    def project(self, x): 
        return self.projection_layer(x) 
    

    
