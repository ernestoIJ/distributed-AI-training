import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
  def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
    super(MNISTModel, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features=input_shape, out_features=hidden_units)
    self.fc2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x