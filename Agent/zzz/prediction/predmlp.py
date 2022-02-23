import os
import pdb

# from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, max_pool


class TrajPredMLP(nn.Module):
    """Predict one feature trajectory, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.LeakyReLU(),
            # nn.Linear(hidden_unit, hidden_unit),
            # nn.LayerNorm(hidden_unit),
            # nn.LeakyReLU(),

            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        # print("in mlp",x, self.mlp(x))
        return self.mlp(x)
