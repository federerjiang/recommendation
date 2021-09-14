import torch.nn as nn
import torch 


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.W(x))

    