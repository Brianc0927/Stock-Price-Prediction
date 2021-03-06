import torch.nn as nn
import torch


class LSTMPredictor(nn.Module):

    def __init__(self, look_back):
        super(LSTMPredictor, self).__init__()

        # Nerual Layers
        self.layer_a = nn.Linear(look_back, 32)
        self.relu    = nn.ReLU()
        self.output  = nn.Linear(32,1)

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input)

    def forward(self, input):

        input = input.view(input.shape[0], input.shape[2], -1)
        logits = self.output(self.relu(self.layer_a(input)))
        logits = logits.view(input.shape[0], logits.shape[2], -1)

        return logits