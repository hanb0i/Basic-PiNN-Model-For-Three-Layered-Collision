
import torch
import torch.nn as nn

class LayerNet(nn.Module):
    def __init__(self, hidden_layers=6, hidden_units=64, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Input: x, y, z (3 coords)
        input_dim = 3
        
        # Optional: Fourier features could be added here
        
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        # Output: u_x, u_y, u_z (3 dims)
        layers.append(nn.Linear(hidden_units, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x shape: (N, 3)
        return self.net(x)

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 Separate networks for 3 layers
        self.layer1 = LayerNet()
        self.layer2 = LayerNet()
        self.layer3 = LayerNet()
        
    def forward(self, x, layer_idx):
        if layer_idx == 0:
            return self.layer1(x)
        elif layer_idx == 1:
            return self.layer2(x)
        elif layer_idx == 2:
            return self.layer3(x)
        else:
            raise ValueError("Invalid layer index")

    def predict_all(self, x):
        # Helper to predict across full domain? 
        # Typically requires knowing which layer x belongs to.
        # For inference, user handles masking.
        pass
