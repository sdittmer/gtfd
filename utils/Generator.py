import torch
import torch.nn as nn
from utils.SineDataSet import get_η

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv0 = nn.Conv1d(1, 4, 3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(4, 4, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(4, 4, 3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose1d(4, 4, 3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose1d(4, 4, 3, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose1d(4, 1, 1, stride=1, padding=0)

        self.linear0 = nn.Linear(4*32, 16)
        self.linear1 = nn.Linear(16, 4*32)

        self.layer_norm0 = nn.LayerNorm([4, 128])
        self.layer_norm1 = nn.LayerNorm([4, 64])
        self.layer_norm2 = nn.LayerNorm([4, 32])
        self.layer_norm3 = nn.LayerNorm([4, 63])
        self.layer_norm4 = nn.LayerNorm([4, 125])

    def forward(self, x):
        res = x.view(-1, 1, 128)

        res = self.conv0(res)
        res = torch.relu(self.layer_norm0(res))

        res = self.conv1(res)
        res = torch.relu(self.layer_norm1(res))

        res = self.conv2(res)
        res = torch.relu(self.layer_norm2(res))

        res = res.view(-1, 128)
        res = torch.relu(self.linear0(res))
        res = torch.relu(self.linear1(res))
        res = res.view(-1, 4, 32)

        res = self.conv3(res)
        res = torch.relu(self.layer_norm3(res))

        res = self.conv4(res)
        res = torch.relu(self.layer_norm4(res))

        res = self.conv5(res)
        res = torch.nn.functional.interpolate(res, size=128, mode='linear', align_corners=True)
        res = res.view(-1, 128)

        return res

    def apply_for_training(self, yδ):
        y_approximation = self(yδ)
        η_approximation = yδ - y_approximation
        y_renoised = y_approximation + get_η(y_approximation.shape)

        return y_approximation, η_approximation, y_renoised