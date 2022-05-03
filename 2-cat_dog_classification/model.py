import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 112
            nn.MaxPool2d(2, 2),  # 56
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 56
            nn.MaxPool2d(2, 2),  # 28
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 28
            nn.MaxPool2d(2, 2),  # 14
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # 14
            nn.MaxPool2d(2, 2),  # 7
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, input):
        x = input['tens']
        x = self.cnn(x)
        output = self.fc(x)
        return output
