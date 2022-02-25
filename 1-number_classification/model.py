import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),  # 28
            nn.MaxPool2d(2, 2),  # 14
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),  # 14
            nn.MaxPool2d(2, 2),  # 7
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*8, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, input):
        x = input['img']
        x = self.cnn(x)
        output = self.fc(x)
        return output