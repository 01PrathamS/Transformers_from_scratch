from embedding import PositionalEncoding, InputEmbedding
from encoder import EncoderBlock, Encoder
from decoder import DecoderBlock, Decoder
from attention import MultiHeadAttentionBlock
from feed_forward import FeedForwardBlock
from transformer import Transformer, ProjectionLayer
import torch
import torch.nn as nn

def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int,
                      d_model: int = 512, 
                      N: int = 6,
                      h: int = 6,
                      d_ff: int = 2048,
                      dropout: float = 0.1) -> Transformer:

    ## create the embedding layer  
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    ## create the positional encoding layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) 
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) 

    ## create the encoder blocks 
    encoder_blocks = [] 
    for _ in range(N): 
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) 
        feed_forward_block = FeedForwardBlock(d_model, d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)  
        encoder_blocks.append(encoder_block)

    ## create the decoder block
    decoder_blocks = []
    for _ in range(N): 
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    ## create the encoder and decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks)) 
    decoder = Decoder(nn.ModuleList(decoder_blocks)) 

    ## create the projection layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) 

    ## create the transformer 
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer) 

    ## Initialize the parameters 
    for p in transformer.parameters(): 
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p) 
        
    return transformer 