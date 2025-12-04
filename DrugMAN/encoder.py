import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention_old(nn.Module):
    #old

    def __init__(self, input_dim, output_dim, n_heads, dropout, device):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = output_dim
        self.n_heads = n_heads
        self.device = device

        assert input_dim % n_heads == 0

        self.w_q = nn.Linear(input_dim, output_dim)
        self.w_k = nn.Linear(input_dim, output_dim)
        self.w_v = nn.Linear(input_dim, output_dim)

        self.fc = nn.Linear(output_dim, output_dim)
        self.do = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        residual = query
        bsz = query.shape[0]
        Q = self.w_q(query).to(self.device)
        K = self.w_k(key).to(self.device)
        V = self.w_v(value).to(self.device)

        Q = Q.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.input_dim // self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2).to(self.device)) / self.scale.to(self.device)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)


        attention = self.do(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.input_dim // self.n_heads))
        x = self.fc(x)

        x += residual
        x = self.layer_norm(x)

        return x

# new

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, dropout, N_tokens, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.N = N_tokens
        self.device = device

        assert output_dim % n_heads == 0
        self.d_head = output_dim // n_heads

        # self.Wq = nn.Parameter(torch.randn(self.N, input_dim, output_dim))
        # self.Wk = nn.Parameter(torch.randn(self.N, input_dim, output_dim))
        # self.Wv = nn.Parameter(torch.randn(self.N, input_dim, output_dim))

        self.Wq = nn.Parameter(torch.empty(self.N, input_dim, output_dim))
        self.Wk = nn.Parameter(torch.empty(self.N, input_dim, output_dim))
        self.Wv = nn.Parameter(torch.empty(self.N, input_dim, output_dim))

        self.bq = nn.Parameter(torch.empty(self.N, output_dim))
        self.bk = nn.Parameter(torch.empty(self.N, output_dim))
        self.bv = nn.Parameter(torch.empty(self.N, output_dim))

        for W in [self.Wq, self.Wk, self.Wv]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

        fan_in = input_dim
        bound = 1 / math.sqrt(fan_in)
        for b in [self.bq, self.bk, self.bv]:
            nn.init.uniform_(b, -bound, bound)


        self.fc = nn.Linear(output_dim, output_dim)
        self.do = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.scale = (self.d_head) ** 0.5

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        N = self.N

        Q = torch.einsum("bti,tio->bto", query, self.Wq) + self.bq
        K = torch.einsum("bti,tio->bto", key, self.Wk) + self.bk
        V = torch.einsum("bti,tio->bto", value, self.Wv) + self.bv

        Q = Q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attn = torch.softmax(scores, dim=-1)
        attn = self.do(attn)

        x_out = attn @ V

        x_out = x_out.transpose(1, 2).reshape(B, N, self.output_dim)

        x_out = self.fc(x_out)

        x_out = self.layer_norm(x_out + query)

        return x_out


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, input_dim, hid_dim, dropout):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hid_dim)
        self.w_2 = nn.Linear(hid_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)

        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, input_dim, output_dim, hid_dim, n_heads, dropout, n_tokens, device):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(input_dim, output_dim, n_heads, dropout, n_tokens, device)
        self.pos_ffn = PositionwiseFeedForward(input_dim, hid_dim, dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, input_dim, output_dim, hid_dim, n_heads, dropout, n_tokens, device):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(input_dim, output_dim, hid_dim, n_heads, dropout, n_tokens, device) for _ in range(n_layers)])

    def forward(self, enc_output):
        enc_output = self.dropout(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output
