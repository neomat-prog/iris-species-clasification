import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisNN(nn.Module):
    def __init__(
        self, input_size=4, hidden_sizes=[8, 16], output_size=3, dropout_rate=0.3
    ):
        super(IrisNN, self).__init__()

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden_layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
                )

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
