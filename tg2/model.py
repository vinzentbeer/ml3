from tinygrad.tensor import Tensor
import tinygrad.nn as nn

class SRCNN:
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def __call__(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x)
        return x

    def parameters(self):
        return [self.conv1.weight, self.conv1.bias, 
                self.conv2.weight, self.conv2.bias,
                self.conv3.weight, self.conv3.bias]