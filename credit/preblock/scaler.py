import torch.nn as nn

try:
    from bridgescaler import load_scaler
except (ImportError, Exception):
    load_scaler = None


class Scaler(nn.Module):
    """
    Scaling layer using a bridgescaler object. Supports transform and its inverse.
    """

    def __init__(self, scaler_path, inverse=False):
        super().__init__()
        self.scaler = load_scaler(scaler_path)
        self.inverse = inverse

    def forward(self, x):
        if self.inverse:
            return self.scaler.inverse_transform(x)
        else:
            return self.scaler.transform(x)
