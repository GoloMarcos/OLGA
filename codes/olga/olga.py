import torch.nn as nn
from torch_geometric.nn import GCNConv


class OLGA(nn.Module):
    def __init__(self, input_len, hidden_lens):
        super(OLGA, self).__init__()

        self.layer1 = GCNConv(input_len, hidden_lens[0])

        self.layer2 = GCNConv(hidden_lens[0], hidden_lens[1])

    def forward(self, x, edge_index):
        h1 = nn.Tanh()(self.layer1(x, edge_index))
        h2 = nn.Tanh()(self.layer2(h1, edge_index))
        return h2



