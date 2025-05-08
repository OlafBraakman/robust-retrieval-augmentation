import torch.nn as nn
from torchvision.ops import MLP

class ClassificationHead(nn.Module):

    def __init__(self, head_config) -> None:
        super().__init__()

        input_dim = head_config['input_dim']
        hidden_dims = head_config['hidden_dims']
        output_dim = head_config['output_dim']
        dropout = head_config['dropout']

        self.layers = nn.Sequential()

        if len(hidden_dims) == 0:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:

            for i in range(len(hidden_dims)):
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        return self.layers(x)
