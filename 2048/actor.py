"""Game Actor of game 2048 """

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        # self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32,4)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.conv_layers(x)
        # x = self.flatten(x)
        x = x.view(-1)
        x = self.fc(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)  # Apply softmax activation along dimension 1
        return x


    
