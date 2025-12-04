import torch
from torch import nn
import torch.nn.functional as F
from DrugMAN.encoder import Encoder


class DrugMAN(nn.Sequential):
    def __init__(self):
        super().__init__()
        # mlp 参数
        self.input_dim_drug = 512
        self.input_dim_protein = 512
        self.dropout = nn.Dropout(0.3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enc_input_dim = self.input_dim_drug
        self.enc_output_dim = self.input_dim_drug
        self.enc_n_heads = 8
        self.enc_hid_dim = 256
        self.enc_n_layers = 5
        self.enc_dropout = 0.3
        self.n_tokens = 4

        self.hidden_dims = [512 * 2, 512, 256, 256]
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug * 2 + self.input_dim_protein * 2] + self.hidden_dims + [1]

        self.encoder = Encoder(self.enc_n_layers, self.enc_input_dim, self.enc_output_dim,
                               self.enc_hid_dim, self.enc_n_heads, self.enc_dropout, self.n_tokens, self.device)

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])
        self.vd2_linear = nn.Linear(768, 512)
        self.vp2_linear = nn.Linear(768, 512)


    def forward(self, vd1, vd2, vp1, vp2):
        vd2 = self.vd2_linear(vd2)
        vp2 = self.vp2_linear(vp2)
        v_dp = torch.stack([vd1, vd2, vp1, vp2], axis=1)
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
