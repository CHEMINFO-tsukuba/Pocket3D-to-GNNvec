import torch
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Set2Set
import torch.nn.functional as F
from torch import Tensor
import numpy as np


#referred to Graphsite-classifier/gnn/model.py
#only jknwm option was choosen.

class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on 
    Graphs with Jumping Knowledge Networks".

    The GNN layers are now MCNWMConv layer
    """

    def __init__(self, num_features, dim, train_eps, num_edge_attr,
                 num_layers, num_channels=1, layer_aggregate='max'):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        self.conv0 = MCNWMConv(in_dim=num_features, out_dim=dim, num_channels=num_channels,
                               num_edge_attr=num_edge_attr, train_eps=train_eps)
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec(
                'self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)'.format(
                    i, num_channels)
            )
            exec('self.bn{} = torch.nn.BatchNorm1d(dim)'.format(i))

        # read out function
        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == 'max':
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == 'mean':
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)

        return x

class NWMConv(MessagePassing):
    """
    The neural weighted message (NWM) layer. output of multiple instances of this
    will produce multi-channel output. 
    """

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr='add')
        self.edge_nn = Sequential(
            Linear(num_edge_attr, 8),
            LeakyReLU(),
            Linear(8, 1),
            ELU()
        )
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr, size=None):
        # x: OptPairTensor
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return '{}(edge_nn={})'.format(self.__class__.__name__, self.edge_nn)


class MCNWMConv(torch.nn.Module):
    """
    Multi-channel neural weighted message module.
    """

    def __init__(self, in_dim, out_dim, num_channels,
                 num_edge_attr=1, train_eps=True, eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(
            Linear(in_dim * num_channels, out_dim), 
            LeakyReLU(), 
            Linear(out_dim, out_dim)
        )
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(
                nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back


        x = self.nn(x)
        self.nn = nn.Sequential(
            nn.Linear(33, 149),  # 入力次元を33に変更
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(149, 149)
        )



        return x

import torch

def vec_converter(node_feature, edge_index, edge_attr):
    embedding_net = JKMCNWMEmbeddingNet(
        num_features    = len(node_feature),
        dim             = len(node_feature),
        train_eps       = True,
        num_edge_attr   = 1,
        num_layers      = 6,
        num_channels    = 3)

    # NumPy配列をtorch.Tensorに変換
    if isinstance(node_feature, np.ndarray):
        node_feature = torch.tensor(node_feature, dtype=torch.float)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # バッチデータの作成
    batch = torch.zeros(node_feature.size(0), dtype=torch.long)

    # フォワードパスの実行
    output = embedding_net(node_feature, edge_index, edge_attr, batch)
    
    # Set2Set 層の出力を取得
    set2set_output = embedding_net.set2set
    print("Set2Set 出力:", set2set_output)
    
    return output

