import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, pool=False, double_channels=False):
        super(ResBlock, self).__init__()
        self.pool = pool
        self.double_channels = double_channels
        self.channels = channels

        self.conv0 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.pooling_layer = nn.AvgPool1d(2)
        self.layer_norm0 = None
        self.layer_norm1 = None

    def forward(self, x):
        if  self.layer_norm0 is None:
            self.layer_norm0 = nn.LayerNorm(x.shape[1:]).cuda()
            self.layer_norm1 = nn.LayerNorm(x.shape[1:]).cuda()
        res = self.conv0(x)
        res = self.layer_norm0(res)
        res = self.leakyrelu(res)
        res = self.conv1(res)
        res = self.layer_norm1(res)
        output = x + res

        if self.pool:
            output = self.pooling_layer(output)
        if self.double_channels:
            output = torch.cat([output, output], 1)

        return output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.cnn_model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            ResBlock( 16, 3, 1, pool=True, double_channels=True),
            ResBlock( 32, 3, 1, pool=True, double_channels=True),
            ResBlock( 64, 3, 1, pool=True, double_channels=True),
            ResBlock(128, 3, 1, pool=True, double_channels=True),
            ResBlock(256, 3, 1, pool=True, double_channels=True),
            ResBlock(512, 3, 1, pool=True),
        )

        self.n_features = 1024

        self.fc_model = nn.Sequential(nn.Linear(self.n_features, 1))

    def forward(self, y):
        y = y.view(-1, 1, 128)
        features = self.cnn_model(y)
        features_flat = features.view(-1, self.n_features)
        validity = self.fc_model(features_flat)

        return validity
