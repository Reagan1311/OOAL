import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class SegDecoder(nn.Module):
    def __init__(
            self,
            embed_dims=512,
            num_layers=3,
            num_heads=8,
    ):
        super().__init__()
        nhead = num_heads
        dim = embed_dims
        # decoder layer
        decoder_layer = DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
        self.decoder = Decoder(decoder_layer, num_layers)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, queries, feat, extra_token=None):
        return self.decoder(queries, feat, extra_token)


class Decoder(TransformerDecoder):
    def forward(self, tgt, memory, extra_token, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        attns = []
        outputs = []
        aff_masks = []
        for mod in self.layers:
            output, attn, aff_mask = mod(output, memory, extra_token, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
            aff_masks.append(aff_mask)

        return outputs, attns, aff_masks


class DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt, memory, extra_token, tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        tgt2, attn2, aff_mask = self.multihead_attn(
            tgt, memory, memory, extra_token)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2, aff_mask


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv, extra_token=None):
        B, Nq, C = xq.size() # 1, 21, 512
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # 16 x 8 x 25 x 256

        aff_mask = None
        if extra_token != None:
            extra_token = extra_token.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            aff_mask = (extra_token @ k.transpose(-2, -1)) * self.scale   # B x 8 x 1 x HW
            aff_mask = torch.sigmoid(aff_mask.mean(dim=1, keepdim=True))   # B x 1 x 1 x HW

        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = aff_mask * attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_save.sum(dim=1) / self.num_heads, aff_mask


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
