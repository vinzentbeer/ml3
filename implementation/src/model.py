from torch import nn


class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv1 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 3, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
