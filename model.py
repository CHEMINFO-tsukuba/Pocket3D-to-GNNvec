# Copyright: Wentao Shi, 2020
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU, ELU
from torch.nn import ModuleList
from torch_geometric.nn import GINConv
from torch_geometric.nn import PNAConv, BatchNorm
from torch_geometric.nn import Set2Set
import itertools
import numpy as np
from torch_geometric.nn import MessagePassing


class GraphsiteClassifier(torch.nn.Module):
    """
    Standard classifier to classify the binding sites.
    """

    def __init__(self, num_classes, num_features, dim, train_eps,
                 num_edge_attr, which_model, num_layers, num_channels,
                 deg=None):
        """
        train_eps: for the SCNWMConv module only when which_model in 
        ['jk', 'residual', 'jknmm', and 'normal'].
        deg: for PNAEmbeddingNet only, can not be None when which_model=='pna'.
        """
        super(GraphsiteClassifier, self).__init__()
        self.num_classes = num_classes

        # use one of the embedding net
        if which_model == 'jknwm':
            self.embedding_net = JKMCNWMEmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr,
                num_layers=num_layers,
                num_channels=num_channels
            )
        else:
            self.embedding_net = EmbeddingNet(
                num_features=num_features,
                dim=dim, train_eps=train_eps,
                num_edge_attr=num_edge_attr)

        # set2set doubles the size of embeddnig
        self.fc1 = Linear(2 * dim, dim)
        self.fc2 = Linear(dim, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_net(
            x=x, edge_index=edge_index,
            edge_attr=edge_attr, batch=batch
        )
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        # returned tensor should be processed by a softmax layer
        return x


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


class EmbeddingNet(torch.nn.Module):
    def __init__(self, num_features, dim, train_eps, num_edge_attr):
        super(EmbeddingNet, self).__init__()

        self.set2set = Set2Set(
            in_channels=dim, processing_steps=5, num_layers=2
        )

        nn1 = Sequential(Linear(num_features, dim),
                         LeakyReLU(), Linear(dim, dim))
        self.conv1 = SCNWMConv(nn1, train_eps, num_features, num_edge_attr)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = SCNWMConv(nn2, train_eps, dim, num_edge_attr)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv3 = SCNWMConv(nn3, train_eps, dim, num_edge_attr)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv4 = SCNWMConv(nn4, train_eps, dim, num_edge_attr)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        #nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv5 = GINConv(nn5)
        #self.bn5 = torch.nn.BatchNorm1d(dim)

        #nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        #self.conv6 = GINConv(nn6)
        #self.bn6 = torch.nn.BatchNorm1d(dim)

        #self.fc1 = Linear(dim, dim)
        # self.fc2 = Linear(dim, dim) # generate embedding here

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)

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

        return x
