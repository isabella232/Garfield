'''Network for training on the Pima Indians Diabetes data.'''
import torch.nn as nn

class PimaNet(nn.Module):
    def __init__(self,num_classes=1):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
                nn.Sigmoid(),
                )

    def forward(self, x):
        return self.net(x)
