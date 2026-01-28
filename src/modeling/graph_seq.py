import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch


class GCN_no_edge_attr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_no_edge_attr, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(MLP, self).__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        return x

class GNN_emb(torch.nn.Module):
    def __init__(self, input_dim, num_layer, emb_dim, edge_attr_option = False, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gcn'):
        super(GNN_emb, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.edge_attr_option = edge_attr_option
        self.JK=JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = nn.Linear(input_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gcn':
                if self.edge_attr_option:
                    self.convs.append(GCNConv(emb_dim, emb_dim))
                else:
                    self.convs.append(GCN_no_edge_attr(emb_dim, emb_dim))
            else:
                raise ValueError(f'Undefined GNN type called {gnn_type}')

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward( self, batched_data ):
        x, edge_index = batched_data.x, batched_data.edge_index
        # 移除了 edge_attr 处理，因为 edge_attr_option=False
        
        x = x.float()
        h_list = [self.atom_encoder(x)]
        
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.sum(torch.stack(h_list, dim=0), dim=0)
        return node_representation