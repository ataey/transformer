import torch
from torch import nn

class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):

        super(PositionEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
