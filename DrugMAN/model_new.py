import torch
import torch
from torch import nn
import torch.nn.functional as F

class DrugMAN(nn.Sequential):

    def __init__(self, custom=False):
        super().__init__()
        self.custom = custom
        if custom:
            self.input_dim_drug = 768
            self.input_dim_protein = 768
        else:
            self.input_dim_drug = 512
            self.input_dim_protein = 512

        self.bilinear = nn.Bilinear(self.input_dim_drug, self.input_dim_protein, 1)

    def forward(self, v_d, v_p):
        print("use ")
        return self.bilinear(v_d, v_p)
