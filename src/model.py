import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union
#AE
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        #编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
            # nn.ReLU()
        )
        #解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoding(self, x):
        return self.encoder(x)

# DAE
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, noise_std=0.1, dropout_rate=0.2):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def add_noise(self, x, noise_std=None):
        if noise_std is None:
            noise_std = self.noise_std
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoding(self, x):
        return self.encoder(x)



class ChannelAttentionSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=1)
        a = self.fc2(F.relu(self.fc1(s), inplace=True))
        a = torch.sigmoid(a).unsqueeze(1)
        return x * a


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        if l > self.max_len:
            raise ValueError(f"Sequence length {l} exceeds max_len {self.max_len}.")
        idx = torch.arange(l, device=x.device)
        pe = self.pos(idx).unsqueeze(0) 
        return x + pe
# class LearnablePositionalEmbedding(nn.Module):
#     def __init__(self, max_len: int, d_model: int):
#         super().__init__()
#         self.pos = nn.Embedding(max_len, d_model)
#         self.prior_proj = nn.Linear(1, d_model, bias=False)  # 固定先验投影
#         self.max_len = max_len
#         im_dict = {
#                 "H5": {
#             "im_site": [119, 123, 125, 126, 127, 129, 138, 140, 141, 151, 152, 153, 154, 155, 156, 185, 189],
#             "antigenic_epitopes_one": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
#             "antigenic_epitopes_two": [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194],
#             "antigenic_epitopes_three": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 256, 257, 258, 259, 260, 261, 262],
#             "antigenic_epitopes_four": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 270, 271, 272, 273, 274, 275, 276, 277],
#             "antigenic_epitopes_five": [197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
#             "loop_130": [130, 131, 132, 133, 134],
#             "helix_190": [186, 187, 188, 189, 190, 191, 192, 193, 194],
#             "loop_220": [217, 218, 219, 220, 221, 222, 223, 224]
#             }
#         }
#         self.importmance_mask = torch.zeros(max_len, dtype=torch.int)
#         for site in im_dict["H5"].keys():
#             for pos in im_dict["H5"][site]:
#                 if pos-1 < max_len:  # 注意序列位置通常是从1开始的，所以要减1
#                     self.importmance_mask[pos-1] = True

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, L, D]
#         importance_mask: [B, L]  (0或1, 固定先验)
#         """
#         b, l, d = x.shape
#         if l > self.max_len:
#             raise ValueError(f"Sequence length {l} exceeds max_len {self.max_len}.")

#         idx = torch.arange(l, device=x.device)
#         pe = self.pos(idx).unsqueeze(0)  # [1, L, D]
#         # 构建[B, L, 1]的importance_mask
#         importance_mask = self.importmance_mask[:l].unsqueeze(0).expand(b, -1).to(x.device)  # [B, L]
#         prior = importance_mask.unsqueeze(-1).float()  # [B, L, 1]
#         # 构建B*L*D的importance_mask

#         prior_emb = prior.expand(-1, -1, d)

#         # prior_emb = self.prior_proj(prior)             # [B, L, D]

#         return x + pe + prior_emb

# class LearnablePositionalEmbedding(nn.Module):
#     def __init__(self, max_len: int, d_model: int):
#         super().__init__()
#         self.pos = nn.Embedding(max_len, d_model)

#         # prior: 0/1 -> D，并配一个可学习强度；从 0 开始更稳
#         self.prior_proj = nn.Linear(1, d_model, bias=False)
#         self.prior_scale = nn.Parameter(torch.tensor(0.0))

#         self.max_len = max_len
#         im_dict = {
#             "H5": {
#                 "im_site": [119, 123, 125, 126, 127, 129, 138, 140, 141, 151, 152, 153, 154, 155, 156, 185, 189],
#                 "antigenic_epitopes_one": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
#                 "antigenic_epitopes_two": [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194],
#                 "antigenic_epitopes_three": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 256, 257, 258, 259, 260, 261, 262],
#                 "antigenic_epitopes_four": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 270, 271, 272, 273, 274, 275, 276, 277],
#                 "antigenic_epitopes_five": [197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
#                 "loop_130": [130, 131, 132, 133, 134],
#                 "helix_190": [186, 187, 188, 189, 190, 191, 192, 193, 194],
#                 "loop_220": [217, 218, 219, 220, 221, 222, 223, 224]
#             }
#         }

#         mask = torch.zeros(max_len, dtype=torch.bool)
#         for site in im_dict["H5"].keys():
#             for pos in im_dict["H5"][site]:
#                 if pos - 1 < max_len:
#                     mask[pos - 1] = True

#         # 用 buffer：自动跟着 .to(device)，也会进 state_dict
#         self.register_buffer("importance_mask", mask)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, l, d = x.shape
#         if l > self.max_len:
#             raise ValueError(f"Sequence length {l} exceeds max_len {self.max_len}.")

#         idx = torch.arange(l, device=x.device)
#         pe = self.pos(idx).unsqueeze(0)  # [1,L,D]

#         prior = self.importance_mask[:l].to(device=x.device).float()  # [L]
#         prior = prior[None, :, None]  # [1,L,1]
#         prior_emb = self.prior_proj(prior)  # [1,L,D]

#         scale = torch.tanh(self.prior_scale)  # 限幅，训练更稳
#         return x + pe + scale * prior_emb
    

class AttnTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    继承 PyTorch TransformerEncoderLayer，强制保留 self-attention 权重到 last_attn:
      last_attn: [B, H, T, T] (batch_first=True 时)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        # 兼容不同 PyTorch 版本（average_attn_weights / is_causal 可能不存在）
        try:
            x_attn, attn = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
        except TypeError:
            # 退化：尽量拿到权重
            x_attn, attn = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )

        # attn 期望是 [B,H,T,T]；若拿到的是 [B,T,T]，补一个 head 维
        if attn is not None and attn.dim() == 3:
            attn = attn.unsqueeze(1)

        self.last_attn = attn
        return self.dropout1(x_attn)


class MyEncoderLayer(nn.Module):
    def __init__(self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        activation: str = "gelu",
        args: object = None,
    ):
        super().__init__()

        # self.encoder_layer1 = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=n_heads,
        #     dim_feedforward=dim_ff,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_first=True,
        #     norm_first=True,)
        # 替换为自定义的 AttnTransformerEncoderLayer，以保留 self-attention 权重
        self.encoder_layer1 = AttnTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        
        self.encoder_layer2 = ChannelAttentionSE(d_model, reduction=8)
        self.args = args
    def forward(self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        # if self.args.model == "no_channel_attn":
        #     return self.encoder_layer1(x, src_key_padding_mask=src_key_padding_mask)
        # else:
        #     return self.encoder_layer2(self.encoder_layer1(x, src_key_padding_mask=src_key_padding_mask))
        return self.encoder_layer1(x, src_key_padding_mask=src_key_padding_mask)
        
class VirusEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        activation: str = "gelu",
        args: object = None,
    ):
        super().__init__()

        encoder_layer = MyEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            dim_ff=dim_ff,
            dropout=dropout,
            activation=activation,
            args=args,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     key_padding_mask: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     return self.enc(x, src_key_padding_mask=key_padding_mask)
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        y = self.enc(x, src_key_padding_mask=key_padding_mask)

        if not return_attn:
            return y

        # 从每一层取出缓存的 last_attn
        attn_list = []
        for layer in self.enc.layers:
            # layer: MyEncoderLayer
            a = getattr(layer.encoder_layer1, "last_attn", None)
            attn_list.append(a)

        return y, attn_list


def masked_mean_max_pool(x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)     
        max_pool = x.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return pooled



@dataclass
class TransformerModelConfig:
    in_channels: int = 64
    d_model: int = 64
    max_len: int = 4096
    enc_layers: int = 2
    enc_heads: int = 4
    enc_ff: int = 256
    dropout: float = 0.1
    ca_reduction: int = 8
    ca_dropout: float = 0.0
    cross_heads: int = 8
    head_hidden: int = 256
    task: str = "regression"
    num_bins: int = 10

class ConvEncoderBlock1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        return y
    
class CNNEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConvEncoderBlock1D(d_model=d_model, kernel_size=kernel_size, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerPairModel(nn.Module):
    def __init__(self, args, cfg: TransformerModelConfig, at_dim: int = 0, sr_dim: int = 0):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.at_dim = at_dim
        self.sr_dim = sr_dim
        
        self.proj1 = nn.Linear(cfg.in_channels, cfg.d_model)
        self.pos1 = LearnablePositionalEmbedding(cfg.max_len, cfg.d_model)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.proj2 = nn.Linear(cfg.in_channels, cfg.d_model)
        self.pos2 = LearnablePositionalEmbedding(cfg.max_len, cfg.d_model)
        self.drop2 = nn.Dropout(cfg.dropout)
        if self.args.model == "cnn":
            self.encoder1 = CNNEncoder(
            d_model=cfg.d_model, n_layers=cfg.enc_layers, kernel_size=15, dropout=cfg.dropout
        )
        else:
            self.encoder1 = VirusEncoder(
                d_model=cfg.d_model, n_layers=cfg.enc_layers, n_heads=cfg.enc_heads,
                dim_ff=cfg.enc_ff, dropout=cfg.dropout, activation="gelu", args=args
        )
        if self.args.model == "cnn":
            self.encoder2 = CNNEncoder(
            d_model=cfg.d_model, n_layers=cfg.enc_layers, kernel_size=15, dropout=cfg.dropout
        )
        else:
            self.encoder2 = VirusEncoder(
                d_model=cfg.d_model, n_layers=cfg.enc_layers, n_heads=cfg.enc_heads,
                dim_ff=cfg.enc_ff, dropout=cfg.dropout, activation="gelu", args=args
            )
        fusion_dim = 4 * cfg.d_model
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, cfg.d_model),
            nn.GELU(), nn.Dropout(cfg.dropout)
        )
        if self.args.model == "no_encoder":
            pooled_dim = 4 * cfg.d_model + at_dim + sr_dim
        elif self.args.model == "no_meta":
            pooled_dim = 2 * cfg.d_model
        else:
            pooled_dim = 2 * cfg.d_model + at_dim + sr_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, cfg.head_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, 1)
        )

        im_dict = {
            "H3": {
                "im_site": [145, 155, 156, 158, 159, 189, 193],
                "antigenic_epitopes_one": [122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146],
                "antigenic_epitopes_two": [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198],
                "antigenic_epitopes_three": [62, 63, 64, 65, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 260, 261, 262, 263, 264, 265],
                "antigenic_epitopes_four": [273, 274, 275, 276, 277, 278, 279, 280, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                "antigenic_epitopes_five": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219],
                "loop_130": [134, 135, 136, 137, 138],
                "helix_190": [190, 191, 192, 193, 194, 195, 196, 197, 198],
                "loop_220": [221, 222, 223, 224, 225, 226, 227, 228]
            },
            "H1": {
                "im_site": [127, 141, 153, 154, 155, 156],
                "antigenic_epitopes_one": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
                "antigenic_epitopes_two": [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170],
                "antigenic_epitopes_three": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 257, 258, 259, 260, 261, 262, 263],
                "antigenic_epitopes_four": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 271, 272, 273, 274, 275, 276, 277, 278],
                "antigenic_epitopes_five": [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
                "loop_130": [131, 132, 133, 134, 135],
                "helix_190": [187, 188, 189, 190, 191, 192, 193, 194, 195],
                "loop_220": [218, 219, 220, 221, 222, 223, 224, 225]
            },
            "H5": {
                "im_site": [119, 123, 125, 126, 127, 129, 138, 140, 141, 151, 152, 153, 154, 155, 156, 185, 189],
                "antigenic_epitopes_one": [115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
                "antigenic_epitopes_two": [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194],
                "antigenic_epitopes_three": [53, 54, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 256, 257, 258, 259, 260, 261, 262],
                "antigenic_epitopes_four": [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 270, 271, 272, 273, 274, 275, 276, 277],
                "antigenic_epitopes_five": [197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
                "loop_130": [130, 131, 132, 133, 134],
                "helix_190": [186, 187, 188, 189, 190, 191, 192, 193, 194],
                "loop_220": [217, 218, 219, 220, 221, 222, 223, 224]
            }
        }
        self.importmance_mask = torch.zeros(cfg.max_len, dtype=torch.int).to(args.device)
        for site in im_dict[args.subtype].keys():
            for pos in im_dict[args.subtype][site]: 
                #序列位置从1开始的，所以要减1
                self.importmance_mask[pos-1] = 1
    # def get_attn_loss(
    #     self,
    #     incoming1: Optional[torch.Tensor],
    #     incoming2: Optional[torch.Tensor],
    #     *,
    #     length: int,
    #     eps: float = 1e-12,
    # ):
    #     capped_attn1 = torch.minimum(
    #         self.importmance_mask * incoming1.mean(dim=0),
    #     torch.tensor(2.0, device=self.importmance_mask.device)
    #     ).sum()
    #     capped_attn2  = torch.minimum(
    #         self.importmance_mask * incoming2.mean(dim=0),
    #     torch.tensor(2.0, device=self.importmance_mask.device)
    #     ).sum()
    #     return  - capped_attn1-capped_attn2
    def get_attn_loss(
        self,
        incoming1,
        incoming2,
        *,
        length: int,
        eps: float = 1e-12,
    ):
        attn1 = incoming1.mean(dim=0)
        attn2 = incoming2.mean(dim=0)

        top_val = 1.8

        # imp1 = attn1 * self.importmance_mask
        # imp2 = attn2 * self.importmance_mask

        # mask1 = (imp1 < 2.0).float()
        # mask2 = (imp2 < 2.0).float()
        mask1 = (attn1 < top_val).int()
        mask2 = (attn2 < top_val).int()
        imp1 = attn1 * self.importmance_mask
        imp2 = attn2 * self.importmance_mask



        loss1 = - (imp1 * mask1).sum()
        loss2 = - (imp2 * mask2).sum()

        return loss1 + loss2
        # return  - (self.importmance_mask * incoming2.mean(dim=0)).sum()-(self.importmance_mask * incoming1.mean(dim=0)).sum()

    def add_pos_for_encoder1(self, x):
        x = self.drop1(self.pos1(self.proj1(x)))
        return x
    
    def add_pos_for_encoder2(self, x):
        x = self.drop2(self.pos2(self.proj2(x)))
        return x
    
    # def forward(self, x, meta_x1, meta_x2):
    #     x1 = x[:, :, 0, :].permute(0, 2, 1)
    #     x2 = x[:, :, 1, :].permute(0, 2, 1)
        
    #     if self.args.model == "no_encoder":
    #         h1 = self.proj1(x1)
    #         h2 = self.proj1(x2)
    #         f = torch.cat([h1, h2], dim=-1)
    #     else:
    #         h1 = self.add_pos_for_encoder1(x1)
    #         h1 = self.encoder1(h1, key_padding_mask=None)
            
    #         h2 = self.add_pos_for_encoder2(x2)
    #         h2 = self.encoder2(h2, key_padding_mask=None)
    #         diff = torch.abs(h1 - h2)
    #         prod = h1 * h2
    #         # 对编码器的表征作增强并融合
    #         f = torch.cat([h1, h2, diff, prod], dim=-1)
    #         f = self.fusion_proj(f)
    
        
    #     u = masked_mean_max_pool(f)
    #     u = torch.cat([u, meta_x1, meta_x2], dim=-1)
        
    #     pred = self.head(u)
    #     return pred
    def forward(self, x, meta_x1, meta_x2, return_attn: bool = False):
        x1 = x[:, :, 0, :].permute(0, 2, 1)
        x2 = x[:, :, 1, :].permute(0, 2, 1)

        attn_info = {}

        if self.args.model == "no_encoder":
            h1 = self.proj1(x1)
            h2 = self.proj1(x2)
            f = torch.cat([h1, h2], dim=-1)
        else:
            h1 = self.add_pos_for_encoder1(x1)
            if return_attn and self.args.model != "cnn":
                h1, attn1 = self.encoder1(h1, key_padding_mask=None, return_attn=True)
                attn_info["attn1_layers"] = attn1
            else:
                h1 = self.encoder1(h1, key_padding_mask=None)

            h2 = self.add_pos_for_encoder2(x2)
            if return_attn and self.args.model != "cnn":
                h2, attn2 = self.encoder2(h2, key_padding_mask=None, return_attn=True)
                attn_info["attn2_layers"] = attn2
            else:
                h2 = self.encoder2(h2, key_padding_mask=None)

            diff = torch.abs(h1 - h2)
            prod = h1 * h2
            f = torch.cat([h1, h2, diff, prod], dim=-1)
            f = self.fusion_proj(f)

        u = masked_mean_max_pool(f)
        if self.args.model != "no_meta":
            u = torch.cat([u, meta_x1, meta_x2], dim=-1)

        pred = self.head(u)

        if not return_attn:
            return pred

        # 默认取“最后一个可用层”的 incoming attention mass
        def _pick_last_attn(attn_layers):
            if not attn_layers:
                return None
            for a in reversed(attn_layers):
                if a is not None:
                    return a
            return None

        if "attn1_layers" in attn_info:
            a1 = _pick_last_attn(attn_info["attn1_layers"])
            if a1 is not None:
                attn_info["incoming1_last"] = incoming_attention_mass(a1, reduce_heads="mean")
        if "attn2_layers" in attn_info:
            a2 = _pick_last_attn(attn_info["attn2_layers"])
            if a2 is not None:
                attn_info["incoming2_last"] = incoming_attention_mass(a2, reduce_heads="mean")

        return pred, attn_info


def incoming_attention_mass(
    attn: torch.Tensor,
    *,
    reduce_heads: str = "mean",
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    attn: [B, H, Tq, Tk] (常见 self-attn: Tq==Tk==T)
    返回: incoming mass [B, Tk]
    """
    if attn.dim() != 4:
        raise ValueError(f"Expected attn [B,H,Tq,Tk], got {tuple(attn.shape)}")

    a = attn

    if query_padding_mask is not None:
        qm = query_padding_mask[:, None, :, None].to(dtype=torch.bool)  # [B,1,Tq,1]
        a = a.masked_fill(qm, 0.0)

    if key_padding_mask is not None:
        km = key_padding_mask[:, None, None, :].to(dtype=torch.bool)  # [B,1,1,Tk]
        a = a.masked_fill(km, 0.0)

    incoming = a.sum(dim=2)  # sum over queries => [B,H,Tk]

    if reduce_heads == "mean":
        return incoming.mean(dim=1)  # [B,Tk]
    if reduce_heads == "sum":
        return incoming.sum(dim=1)   # [B,Tk]
    raise ValueError("reduce_heads must be 'mean' or 'sum'")