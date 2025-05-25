import torch
import torch.nn as nn

class SemanticBiasLayer(nn.Module):
    def __init__(self, dim_weight_map):
        super().__init__()
        self.bias = torch.tensor([
            dim_weight_map.get(i, 1.0) for i in range(384)
        ], dtype=torch.float32)

    def forward(self, x):
        return x * self.bias


class SemanticMind(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            SemanticBiasLayer({
                10: 1.4, 370: 1.3  # эмоция
            }),
            SemanticBiasLayer({
                322: 1.2, 284: 1.1  # человечность
            }),
            SemanticBiasLayer({
                95: 0.9, 229: 0.8   # научная строгость подавлена
            })
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
