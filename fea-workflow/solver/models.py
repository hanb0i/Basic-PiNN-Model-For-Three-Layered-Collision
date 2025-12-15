
import torch
import torch.nn as nn

class LayerNet(nn.Module):
    def __init__(self, hidden_layers=6, hidden_units=64, activation=nn.Tanh()):
        super().__init__()
        layers = []
        input_dim = 3
        
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        layers.append(nn.Linear(hidden_units, 3))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.net(x)

class MultiLayerPINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        depth = config['pinn']['layers']['hidden_depth']
        width = config['pinn']['layers']['hidden_width']
        
        self.layer1 = LayerNet(hidden_layers=depth, hidden_units=width)
        self.layer2 = LayerNet(hidden_layers=depth, hidden_units=width)
        self.layer3 = LayerNet(hidden_layers=depth, hidden_units=width)
        
    def forward(self, x, layer_idx):
        if layer_idx == 0:
            return self.layer1(x)
        elif layer_idx == 1:
            return self.layer2(x)
        elif layer_idx == 2:
            return self.layer3(x)
        raise ValueError("Invalid layer index")
