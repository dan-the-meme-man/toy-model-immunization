import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input channels to 16 output channels with 5x5 convolution
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operations: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 16 channels each of size 5x5 to 1x120 vector
        self.fc2 = nn.Linear(120, 84) # 1x120 vector to 1x84 vector
        self.fc3 = nn.Linear(84, 10) # 1x84 vector to 1x10 vector

    def forward(self, x):
        # average pooling over a 2x2 window
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), 2, padding=2)
        # If the size is a square, you can specify with a single number
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.sigmoid(self.fc1(x)) # linear + sig activation
        x = F.sigmoid(self.fc2(x)) # linear + sig activation
        x = self.fc3(x) # linear
        return x