import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolutionWithResidual(nn.Module):
    """
    GCNII Layer, https://github.com/chennnM/GCNII
    """

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolutionWithResidual, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(
            self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class TupleAdj_GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(TupleAdj_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, pd, pt):
        # adj = P D Pt: PD = pd, Pt = pt
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        midpu = torch.spmm(pt, support)
        output = torch.spmm(pd, midpu)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConv4MulChannel(Module):
    """
    # 在每个节点特征基础上增加一个随机的emb，形成上下两层的双通道特征张量
    # 双通道的特征经过双通道的卷积之后变成单通道的特征，故每层需有两个双通道的卷积核，双通道卷积核可用一个Tensor来存储和操作
    """

    def __init__(self, in_features, out_features, channel, bias=False):
        super(GraphConv4MulChannel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channel = channel
        self.weight = Parameter(torch.FloatTensor(
            channel*channel, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(channel, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        channel_weight = torch.split(self.weight, self.channel, dim=0)
        # adj.to_dense()不是稀疏张量的话，不用转换成稠密的。在新建维度上堆积
        adj = torch.stack((adj, adj), dim=0)
        output = None
        for i in range(self.channel):
            support = torch.matmul(input, channel_weight[i])
            mid = torch.matmul(adj, support)
            mid = torch.sum(mid, dim=0)
            if output is None:
                output = mid
            else:
                # 在已有维度上级联 torch.cat
                output = torch.stack((output, mid), dim=0)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'
