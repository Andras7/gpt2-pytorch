import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_


class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_dim, drop=0.1):
        super().__init__()
        self.temperature = np.power(key_dim, 0.5)
        self.dropout = nn.Dropout(drop)

    def forward(self, q, k, v):
        energies = (torch.bmm(q, k.transpose(1, 2))) / self.temperature

        seq_len = energies.size(1)
        mask = (torch.tril(torch.ones(seq_len, seq_len)) == 0).to(energies.device)
        energies.masked_fill_(mask, -np.inf)

        attention = self.dropout(F.softmax(energies, dim=2))
        context = torch.bmm(attention, v).squeeze(1)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = num_heads
        self.attn = ScaledDotProductAttention(embed_dim)

        self.query_trans = nn.Linear(embed_dim, embed_dim)
        self.keys_trans = nn.Linear(embed_dim, embed_dim)
        self.value_trans = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        kaiming_normal_(self.query_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.keys_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.value_trans.weight, nonlinearity="linear")
        kaiming_normal_(self.projection.weight, nonlinearity="linear")

        self.keys_trans = self.keys_trans.half()
        self.value_trans = self.value_trans.half()
        self.projection = self.projection.half()

    def split_heads(self, x):
        bs, seq_len, _ = x.size()
        # result: (head*batch_size) x seq_len x new features
        return x.view(bs, seq_len, self.n_heads, -1).permute(2, 0, 1, 3).reshape(self.n_heads * bs, seq_len, -1)

    def merge_heads(self, x):
        _, seq_len, features_size = x.size()
        x = x.view(self.n_heads, -1, seq_len, features_size)
        bs = x.size(1)
        # batch_size x seq_len x heads x features
        return x.permute(1, 2, 0, 3).reshape(bs, seq_len, -1)

    def forward(self, x):
        q = self.split_heads(self.query_trans(x))
        k = self.split_heads(self.keys_trans(x.half()).float())
        v = self.split_heads(self.value_trans(x.half()).float())
        a, _ = self.attn(q, k, v)
        a = self.merge_heads(a)
        return self.projection(a.half())


class MLP(nn.Module):
    def __init__(self, embed_dim, factor=4):
        super(MLP, self).__init__()
        self.fc = nn.Linear(embed_dim, embed_dim * factor)
        self.fc2 = nn.Linear(embed_dim * factor, embed_dim)

        kaiming_normal_(self.fc.weight, nonlinearity="relu")
        kaiming_normal_(self.fc2.weight, nonlinearity="linear")

        self.fc = self.fc.half()
        self.fc2 = self.fc2.half()

    def forward(self, x):
        h = F.gelu(self.fc(x.half()).float())
        return self.fc2(h.half())


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def get_positional_encoding(n_positions, n_embd):
    def angle_defn(pos, i, d_model_size):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model_size))
        return pos * angle_rates

    angle_rads = angle_defn(np.arange(n_positions)[:, np.newaxis], np.arange(n_embd)[np.newaxis, :], n_embd)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = torch.tensor(np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...], dtype=torch.float)
    return pos_encoding


class GPT2Model(nn.Module):
    def __init__(self):
        super(GPT2Model, self).__init__()
        self.n_layer = 12
        self.n_embd = 768 + 128
        self.n_head = 12 + 2
        self.n_tokens = 512
        self.n_positions = 512  # TODO

        self.wte = nn.Embedding(self.n_tokens, self.n_embd, padding_idx=0).half()
        self.register_buffer('positional_encoding', get_positional_encoding(self.n_positions, self.n_embd))
        self.blocks = nn.ModuleList([Block(self.n_embd, self.n_head) for _ in range(self.n_layer)])

        self.decoder_norm = nn.LayerNorm(self.n_embd)
        self.decoder = nn.Linear(self.n_embd, self.n_tokens, bias=False)
        kaiming_normal_(self.decoder.weight, nonlinearity="linear")

    def forward(self, input_ids):
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.positional_encoding[:, :input_ids.size(1), :]
        hidden_states = (inputs_embeds + position_embeds).float()

        for block in self.blocks:
            hidden_states = block(hidden_states)

        decoded = self.decoder(self.decoder_norm(hidden_states))
        return F.log_softmax(decoded, dim=-1)
