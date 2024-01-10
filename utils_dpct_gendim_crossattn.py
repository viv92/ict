### Utilities implementing Transformer backbone for a diffusion model
## Features:
# 1. From the original DiT Paper, the transformer backbone for diffusion model is not supposed to have a causal mask
# 2. This implementation just implements the non-causal decoder with input = [t, noised_x] and output = [denoised_x]. No separate 'final_emb' used.
# 3. For the condition_emb, we use an external encoder (e.g. t5_encoder for text modality). The condition_emb is fed to the decoder via xattn. 
# 4. Don't use sinusoidal embeddings for time since time here is diffusion time
# 5. Decoder layer is modified to skip xattn layer if condition_emb is None (for CFG)


import torch
import torch.nn as nn
from copy import deepcopy 


## Transformer Modules 

# utility function to create N copies of a module as a list (note: not sequential)
def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# utility function to create upper triangular mask for decoder masked attention
def subsequent_mask(mask_shape):
    batch_size, max_seq_len = mask_shape
    mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).type(torch.uint8) # mask.shape = [max_seq_len, max_seq_len] - this is the expected mask.shape for causal mask for nn.MultiheadAttention module
    return mask == 1  # True elements are masked

# utility function to create mask over pad tokens
def pad_mask(keys, pad_token):
    batch_size, max_seq_len = keys.shape
    mask = keys.eq(pad_token) # mask.shape: [batch_size, max_seq_len] - this is the expected mask.shape for padding mask for nn.MultiheadAttention module
    return mask  # True elements are masked

# class implementing the feed forward block (used for each encoder / decoder layer - after the multihead attention block)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.GELU()
    def forward(self, x):
        return self.w2(self.dropout( self.act_fn(self.w1(x)) ))

# class implementing multi head attention (using pytorch inbuilt nn.MultiheadAttention)
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout, kdim=d_k*n_heads , vdim=d_v*n_heads , batch_first=True) # kdim is the total dim (multiplying with n_heads)
    # function to calculate (masked or unmasked) multihead attention
    def forward(self, key, query, value, mask_padding=None, mask_causal=None): # can be used for both (unmasked) encoder attention and (masked) decoder attention
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask=mask_padding, attn_mask=mask_causal)
        return attn_output

# class implementing residual + normalization connection - takes in any block and applies a normalization + residual connection
class SublayerConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer): # sublayer can be any functional block
        return x + self.dropout(sublayer( self.norm(x) )) # note that we apply the norm first

# NOTE transformer encoder not used 
# # class implementing a single encoder layer
# # each encoder layer has two blocks: 1. (self) multihead attention 2. feed_forward; with sublayer connection around each
# class EncoderLayer(nn.Module):
#     def __init__(self, self_attn, feed_forward, dim, dropout):
#         super().__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayers = clones(SublayerConnection(dim, dropout), 2) # one for self_attn block and other for feed_forward block
#     def forward(self, x, mask_padding, mask_causal):
#         x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask_padding=mask_padding, mask_causal=mask_causal)) # x.shape: [batch_size, seq_len, d_model]
#         x = self.sublayers[1](x, self.feed_forward) # x.shape: [batch_size, seq_len, d_model]
#         return x

# # class implementing the entire encoder block = stacked encoder layers
# class Encoder(nn.Module):
#     def __init__(self, layer, N, d_model):
#         super().__init__()
#         self.layers = clones(layer, N)
#         self.norm = nn.LayerNorm(d_model) # final layernorm at encoder output
#     def forward(self, x, mask_padding=None, mask_causal=None):
#         for layer in self.layers:
#             x = layer(x, mask_padding, mask_causal)
#         return self.norm(x)

# class implementing a single decoder layer
# each decoder layer has three blocks: 1. (self) (masked) multihead attention 2. (src) (unmasked) multihead x-attention  3. feed_forward; with sublayer connection around each
# NOTE Decoder layer is modified to skip xattn layer if condition_emb is None (for CFG)
class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(dim, dropout), 3) # one for self_attn block, second for src_attn block, third for feed_forward block
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        m = encoder_out
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask_padding=tgt_mask_padding, mask_causal=tgt_mask_causal)) # first apply self_attn block
        if m is not None:
            x = self.sublayers[1](x, lambda x: self.src_attn(m, x, m, mask_padding=src_mask_padding)) # src_attn: (key from encoder, query from decoder, value from encoder)
        x = self.sublayers[2](x, self.feed_forward)
        return x

# class implementing the entire decoder block = stacked decoder layers
class Decoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(d_model) # final layernorm at decoder output
    def forward(self, x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask_padding, tgt_mask_padding, tgt_mask_causal)
        return self.norm(x)


## Custom causal DiT Modules

# class to add positional encoding to embeddings (note that this class implements positional encoding as a constant untrainable vector)
class PositionalEncoding_Fixed(nn.Module):
    def __init__(self, d_model, maxlen):
        super().__init__()
        # calculate positional encoding and save them (register buffer for saving params in params_dict that are not to be updated during backprop)
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1) # pos.shape: [maxlen, 1]
        div_term = 10000.0 * torch.exp( torch.arange(0, d_model, 2) / d_model ) # div_term.shape: [d_model/2]
        pe[:, 0::2] = torch.sin(pos / div_term) # pe[:, 0::2].shape: [maxlen, d_model/2]
        pe[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer("pe", pe)
    # add positional encoding to the embedding - and freeze positional encoding
    def forward(self, x): # x.shape: [batch_size, seq_len, d_model]
        batch_size, seq_len = x.shape[0], x.shape[1]
        pos_emb = self.pe[:seq_len, :].requires_grad_(False) # [seq_len, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # [b, seq_len, d_model]
        x = x + pos_emb 
        return x 


class DPCT_Embedder(nn.Module):
    def __init__(self, max_seq_len_dpct, x_seq_len, d_model, x_dim, dropout, pos_enc, condition_dim, device):
        super().__init__()
        self.x_emb = nn.Linear(x_dim, d_model)
        self.condition_emb = nn.Linear(condition_dim, d_model)
        self.t_emb = nn.Linear(1, d_model)
        self.final_emb = nn.Parameter(torch.ones(x_seq_len, d_model)) # learnable embedding for placeholder <final> token sequence
        self.pos_enc = pos_enc
        self.pos_emb_learnt = nn.Parameter(torch.randn(max_seq_len_dpct, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) # NOTE: dropout acts on channel dim, not on seq len
        self.d_model = d_model
        self.max_seq_len_dpct = max_seq_len_dpct # x_seq_len + 1 (for time)
        self.x_seq_len = x_seq_len
        self.device = device

    # function to get time embeddings from time int (based on sinusoidal position encoding)
    # NOTE sinusoidal embeddings not used for diffusion time
    # def get_time_embedding(self, t, d_model): # t.shape: [batch_size]
    #     t = t.unsqueeze(-1).float()
    #     inv_freq = 1.0 / (
    #         10000
    #         ** (torch.arange(0, d_model, 2, device=self.device).float() / d_model)
    #     )
    #     pos_enc_a = torch.sin(t.repeat(1, d_model // 2) * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(1, d_model // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc

    # function for forward prop 
    def forward(self,
                x,  # x.shape: [batch_size, x_seq_len, x_dim]
                t,  # t.shape: [batch_size]
                condition_emb # shape: [batch_size, condition_seq_len, condition_dim]
                ):     
        batch_size = x.shape[0]
        x_emb = self.x_emb(x)
        if condition_emb is not None: # cfg
            condition_emb = self.condition_emb(condition_emb) 
        t_emb = self.t_emb(t.unsqueeze(-1)) # t_emb.shape: [batch_size, d_model]
        t_emb = t_emb.unsqueeze(1) # t_emb.shape: [batch_size, 1, d_model]
        final_emb = self.final_emb.unsqueeze(0).expand(batch_size, -1, -1) # final_emb.shape: [batch_size, x_seq_len, d_model]
        dit_input_emb = torch.cat([t_emb, x_emb, final_emb], dim=1) # dit_input_emb.shape: [batch_size, max_seq_len_dpct, d_model]
        pos_emb_learnt = self.pos_emb_learnt.unsqueeze(0).expand(batch_size, -1, -1)

        # add positional encoding 
        # dit_input_emb = self.pos_enc(dit_input_emb)
        dit_input_emb += pos_emb_learnt

        dit_input_emb = self.dropout( self.norm(dit_input_emb) ) # NOTE: dropout acts on channel dim, not on seq len
        return dit_input_emb, condition_emb 



# class implementing the dit transformer (non-causal) constituting the diffusion backbone
class DPCT(nn.Module):
    def __init__(self, embedder, decoder, max_seq_len_dpct, x_seq_len, d_model, x_dim, device):
        super().__init__()
        self.embedder = embedder
        self.decoder = decoder
        self.max_seq_len_dpct = max_seq_len_dpct # x_seq_len * 2 + 1 (for time)
        self.x_seq_len = x_seq_len 
        self.d_model = d_model
        self.final_proj = nn.Linear(d_model, x_dim*1) # denoised output 
        self.device = device
        self.mask_causal = self.make_causal_mask(max_seq_len_dpct).to(device)

    def make_causal_mask(self, max_seq_len_dpct):
        mask = torch.triu(torch.ones(max_seq_len_dpct, max_seq_len_dpct), diagonal=1).type(torch.uint8)
        return mask == 1  # True elements are masked

    def forward(self, x, # x.shape: [b, x_seq_len, x_dim] 
                    t, # diffusion time t.shape: [b]
                    condition_emb # shape: [b, condition_seq_len, condition_dim] # NOTE condition_emb is projected from condition_dim to d_model
                ):
        dpct_input_emb, condition_emb = self.embedder(x, t, condition_emb) # dpct_input_emb.shape: [batch_size, max_seq_len_dpct, d_model]
        dpct_out_seq = self.decoder(dpct_input_emb, encoder_out=condition_emb, src_mask_padding=None, tgt_mask_padding=None, tgt_mask_causal=self.mask_causal) # dpct_out_seq.shape: [batch_size, max_seq_len_dpct, d_model]
        x_denoised = dpct_out_seq[:, -self.x_seq_len:] # x_denoised.shape: [batch_size, x_seq_len, d_model]
        x_denoised = self.final_proj(x_denoised) # x_denoised.shape: [batch_size, x_seq_len, x_dim * 1]
        return x_denoised # the predicted denoised x 



# caller function to init dpct
def init_dpct(max_seq_len_dpct, x_seq_len, d_model, x_dim, condition_dim, d_k, d_v, n_heads, n_layers, d_ff, dropout, device):
    assert max_seq_len_dpct == x_seq_len * 2 + 1
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    decoder_layer = DecoderLayer(deepcopy(attn), deepcopy(attn), deepcopy(ff), d_model, dropout) # single decoder layer
    decoder = Decoder(decoder_layer, n_layers, d_model) # decoder = stacked decoder layers
    positional_encoder_fixed = PositionalEncoding_Fixed(d_model, max_seq_len_dpct)
    dit_embedder = DPCT_Embedder(max_seq_len_dpct, x_seq_len, d_model, x_dim, dropout, positional_encoder_fixed, condition_dim, device)
    model = DPCT(dit_embedder, decoder, max_seq_len_dpct, x_seq_len, d_model, x_dim, device)
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

