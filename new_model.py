import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUNetwork(nn.Module):
    def __init__(self):
        super(GRUNetwork, self).__init__()
        
        # Hidden size inferred from GRU output: 322
        self.hidden_size = 322
        self.output_size = 161
        
        # GRU 1 and GRU 2 with input size 322 (matches the input feature size)
        self.gru1 = nn.GRU(input_size=322, hidden_size=322, batch_first=True)
        self.gru2 = nn.GRU(input_size=322, hidden_size=322, batch_first=True)
        
        # Final linear layer: 322 -> 161
        self.fc = nn.Linear(322, 161)

    def forward(self, x, h01, h02):
        # x: (batch, seq_len=1, features=322)
        # h01, h02: (1, batch, 322)
        
        out1, _ = self.gru1(x, h01)   # out1: (batch, 1, 322)
        out1 = out1.squeeze(1)        # (batch, 322)
        
        h02 = h02.squeeze(0)          # (batch, 322)
        out2_input = out1 + h02       # residual connection
        
        out2_input = out2_input.unsqueeze(1)  # (batch, 1, 322)
        out2, _ = self.gru2(out2_input)       # (batch, 1, 322)
        out2 = out2.squeeze(1)                # (batch, 322)
        
        logits = self.fc(out2)                # (batch, 161)
        out = torch.sigmoid(logits)           # (batch, 161)
        out = torch.clamp(out, 0.0, 1.0)      # Clip layer (optional if sigmoid used)
        
        return out


model = GRUNetwork()

batch_size = 4
x = torch.randn(batch_size, 1, 322)
h01 = torch.randn(1, batch_size, 322)
h02 = torch.randn(1, batch_size, 322)

output = model(x, h01, h02)
#print(output.shape)  # (batch_size, 161)
