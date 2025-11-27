import torch
from torch import nn
import torch.nn.functional as F
from DrugMAN.encoder import Encoder


class DrugMAN(nn.Sequential):
    def __init__(self, custom=False):
        super(DrugMAN, self).__init__()
        # mlp 参数
        self.custom = custom
        if custom:
            self.input_dim_drug = 768
            self.input_dim_protein = 768
        else:
            self.input_dim_drug = 512
            self.input_dim_protein = 512
        self.dropout = nn.Dropout(0.3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # encoder 参数
        self.enc_input_dim = self.input_dim_drug  # self_attn的参数
        self.enc_output_dim = self.input_dim_drug   # self_attn的参数
        self.enc_n_heads = 8  # self_attn的参数
        self.enc_hid_dim = 256  # PositionwiseFeedForward的参数
        self.enc_n_layers = 5
        self.enc_dropout = 0.3

        self.hidden_dims = [512, 256, 256]  # [512,256,256]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]

        self.encoder = Encoder(self.enc_n_layers, self.enc_input_dim, self.enc_output_dim,
                               self.enc_hid_dim, self.enc_n_heads, self.enc_dropout, self.device)

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

        if custom:
            # self.Wpr = nn.Sequential(
            #     nn.Linear(15599, 8000),
            #     nn.ReLU(),
            #     nn.Linear(8000, 2048),
            #     nn.ReLU(),
            #     nn.Linear(2048, 768),
            #     nn.ReLU()
            # )
            self.Wpr = nn.Sequential(
                nn.Linear(15599, 768),
                nn.ReLU(),
            )

    def forward(self, v_d, v_p):
        if self.custom:
            v_p = self.Wpr(v_p)
        v_dp = torch.stack([v_d, v_p], axis=1)
        bs = v_dp.shape[0]
        v_f = self.encoder(v_dp)    # [bcs, 2,512]
        v_f = v_f.view(bs, 1, -1)  # [bcs, 1, 1024]
        v_f = torch.squeeze(v_f)     # [bcs, 1024]
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = self.dropout(F.relu(l(v_f)))
        return v_f
