import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, num_feats, num_nodes,
                 hidden1, hidden2, dropout):
        super(GCNModelVAE, self).__init__()

        self.input_dim = num_feats
        self.n_samples = num_nodes
        self.num_out = hidden2
        self.hidden = GraphConvolution(num_feats, hidden1)
        self.act = nn.ReLU()
        self.l_z_mean = GraphConvolution(hidden1, hidden2)
        self.l_log_std = GraphConvolution(hidden1, hidden2)

        # self.decoder = InnerProductDecoder(input_dim=hidden2,
        #                                    act=lambda x: x)
        self.decoder = InnerProductDecoder(input_dim=hidden2,
                                           act=nn.Sigmoid())

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.hidden.reset_parameters()
        self.l_z_mean.reset_parameters()
        self.l_log_std.reset_parameters()

    def forward(self, x, adj):
        # layer hidden
        h = F.dropout(x, self.dropout, training=self.training)
        h = self.hidden(h, adj)
        h = self.act(h)

        # layer z_mean
        h1 = F.dropout(h, self.dropout, training=self.training)
        self.z_mean = self.l_z_mean(h1, adj)

        # layer log_std
        h2 = F.dropout(h, self.dropout, training=self.training)
        self.log_std = self.l_log_std(h2, adj)

        z = self.z_mean + torch.randn(self.n_samples,
                                      self.num_out) * torch.exp(self.log_std)

        reconstructions = self.decoder(z)
        return reconstructions


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim, dropout=0, act=nn.Sigmoid(),):
        super(InnerProductDecoder, self).__init__()

        self.input_dim = input_dim
        self.dropout = dropout
        self.act = act

    def forward(self, x):
        h = F.dropout(x, self.dropout, training=self.training)
        # h = torch.mm(h, h.T).reshape(-1)
        h = torch.mm(h, h.T)
        output = self.act(h)
        return output
