import torch
# 1. Create LinearClassifier Class
# 2. Process Data

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=1000, output_dim=2000):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)

  def forward(self, x):
    x = self.linear(x)
    return x
  
  