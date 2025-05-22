import torch.nn as nn
import torch.nn.functional as F
import torch
import random

from models.layers import GraphConvolution, TupleAdj_GraphConvolution, GraphConv4MulChannel, GraphConvolutionWithResidual
from models.mlp import MLP, MLP_v2


class Mod_SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass):
        super(Mod_SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        # x = adj^k * X； none_adj没用只是为了和其他的方法调用参数列表保持一致
        x = self.W(x)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        # return self.W(x) # 文章中是softmax；但源码中没有softmax，外边调用下面接交叉熵函数
        x = self.W(x)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, nlayer, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.nlayer = nlayer
        self.GCNlayers = torch.nn.ModuleList()
        self.GCNlayers.append(GraphConvolution(nfeat, nhid))
        for _ in range(nlayer - 2):
            self.GCNlayers.append(GraphConvolution(nhid, nhid))
        self.GCNlayers.append(GraphConvolution(nhid, nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        for i in range(self.nlayer - 1):
            x = F.relu(self.GCNlayers[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.GCNlayers[-1](x, adj)
        return F.log_softmax(x, dim=1)


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolutionWithResidual(
                nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(
                layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i+1))
        layer_inner = F.dropout(
            layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class FCN(nn.Module):
    def __init__(self, layers_config, dropout, res=False):
        super(FCN, self).__init__()
        # layers_config：list 有层数加1个元素，连续的两个元素表示了单层输入和输出的维度

        # 利用维度初始化每层的卷积操作
        self.GCNlayers = torch.nn.ModuleList()
        self.Reslayers = torch.nn.ModuleList()
        for From, To in zip(layers_config[:-1], layers_config[1:]):
            self.GCNlayers.append(GraphConvolution(From, To))
            # self.GCNlayers.append(TupleAdj_GraphConvolution(From, To)) # 在近似数目与原来节点差距不大时内存会占用的更多，几乎是2倍
            if res:
                self.Reslayers.append(nn.Linear(From, To))
        # 利用层数初始化激活函数
        self.activationList = list()
        for i in range(len(layers_config) - 2):
            self.activationList.append(F.relu)
        self.activationList.append(F.log_softmax)  # F.softmax()

        self.dropout = dropout
        self.res = res

    #pd, pt
    def forward(self, x, adj):  # adj_list均被拆解为pd_list, pt_list
        iter_num = 0
        # adj_list len(pd_list) == len(pt_list)
        # assert len(adj) == len(self.GCNlayers)
        # 中间层
        while iter_num < len(self.GCNlayers) - 1:
            # adj_list pd_list[iter_num], pt_list[iter_num]
            if self.res:
                x_pre = self.Reslayers[iter_num](x)
                x = self.GCNlayers[iter_num](x, adj)
                x = x + x_pre
            else:
                x = self.GCNlayers[iter_num](x, adj)
                x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            iter_num += 1
        # 输出层
        # adj_list pd_list[iter_num], pt_list[iter_num]
        if self.res:
            x_pre = self.Reslayers[iter_num](x)
            x = self.GCNlayers[iter_num](x, adj)
            x = x + x_pre  # final residual layer
        else:
            x = self.GCNlayers[iter_num](x, adj)
        out = F.log_softmax(x, dim=1)  # F.log_softmax(x, dim=1)

        return out


class DAGNN(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, dropout, simple=False):
        super(DAGNN, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer
        self.mlp = MLP_v2(
            num_mlp_layers, nfeat, mlp_hidden_dim, nclass, dropout)
        self.proj = nn.Linear(nclass, 1)

        self.dropout = dropout
        self.simple = simple

    def forward(self, x, adj):
        # h = x.to_dense()
        h = x
        h = self.mlp(h)
        hidden_rep = []
        for i in range(self.layer_num):
            # h = torch.spmm(adj, h)
            h = torch.mm(adj, h)
            hidden_rep.append(h)

        if self.simple:
            return F.log_softmax(h)
        else:
            h_rep = torch.stack(hidden_rep, dim=1)
            retain_score = self.proj(h_rep)
            retain_score = retain_score.squeeze()

            retain_score = torch.sigmoid(retain_score)
            retain_score = retain_score.unsqueeze(1)
            out = torch.matmul(retain_score, h_rep).squeeze()
            out = F.log_softmax(out)
            return out


class JKNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 mode='cat',
                 out_message_passing=True,
                 dropout=0.):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(in_dim, hid_dim))
        for _ in range(num_layers):
            self.layers.append(GraphConvolution(hid_dim, hid_dim))

        if self.mode == 'cat':
            hid_dim = hid_dim * (num_layers + 1)
        elif self.mode == 'lstm':
            self.lstm = nn.LSTM(hid_dim, (num_layers * hid_dim) //
                                2, bidirectional=True, batch_first=True)
            self.attn = nn.Linear(2 * ((num_layers * hid_dim) // 2), 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.out_message_passing = out_message_passing
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.attn.reset_parameters()

    def forward(self, x, adj):
        feat_lst = []
        feats = x
        for layer in self.layers:
            feats = self.dropout(layer(feats, adj))
            feat_lst.append(feats)

        if self.mode == 'cat':
            out = torch.cat(feat_lst, dim=-1)
        elif self.mode == 'max':
            out = torch.stack(feat_lst, dim=-1).max(dim=-1)[0]
        else:
            # lstms
            x = torch.stack(feat_lst, dim=1)
            alpha, _ = self.lstm(x)
            alpha = self.attn(alpha).squeeze(-1)
            alpha = torch.softmax(alpha, dim=-1).unsqueeze(-1)
            out = (x * alpha).sum(dim=1)

        # # DGL
        # g.ndata['h'] = out
        # g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

        h = out
        if self.out_message_passing:
            # h = torch.spmm(adj.T, h)
            h = torch.mm(adj.T, h)

        return F.log_softmax(self.output(h), dim=-1)


class FCN_v5(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, dropout, proj_type='linear', res=False):
        super(FCN_v5, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer

        self.mlp_1 = MLP(
            num_mlp_layers, nfeat, mlp_hidden_dim, mlp_hidden_dim)
        self.mlp_2 = MLP(
            num_mlp_layers, mlp_hidden_dim, mlp_hidden_dim, nclass)

        self.proj_in = nn.Linear(nfeat, mlp_hidden_dim)
        self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        self.proj_type = proj_type
        self.res = res
        self.dropout = dropout

    def forward(self, x, adj):
        h = x
        if self.proj_type == "mlp":
            h = self.mlp_1(h)
        else:
            h = self.proj_in(h)
            h = F.relu(h)

        pre_h = h
        hidden_rep = []
        for i in range(self.layer_num):
            # h = torch.spmm(adj, h)
            h = torch.mm(adj, h)
            if self.res:
                h = h + pre_h
                pre_h = h
            h = F.dropout(h, self.dropout, training=self.training)
            hidden_rep.append(h)

        score_over_layer = 0.
        for h in hidden_rep:
            score_over_layer += h

        h = self.proj_out(score_over_layer)
        out = F.log_softmax(h)
        return out


class FCN_v6(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, dropout,
                 proj_type='linear', out_type="direct", res=False):
        super(FCN_v6, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer

        if proj_type == 'linear':
            self.proj_in = nn.Linear(nfeat, mlp_hidden_dim)
        else:
            self.proj_in = MLP_v2(
                num_mlp_layers, nfeat, mlp_hidden_dim, mlp_hidden_dim, dropout)

        if out_type == "direct" or out_type == "sum":
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)
        elif out_type == "stack":
            self.proj_out = nn.Linear(mlp_hidden_dim * (nlayer + 1), nclass)
        elif out_type == "mlp" or out_type == "mlp_sum":
            self.proj_out = MLP_v2(
                num_mlp_layers, mlp_hidden_dim, mlp_hidden_dim, nclass, dropout)
        elif out_type == "mlp_stack":
            self.proj_out = MLP_v2(
                num_mlp_layers, mlp_hidden_dim * (nlayer + 1), mlp_hidden_dim, nclass, dropout)
        else:
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        self.lin1 = nn.Linear(nfeat, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)

        self.params1 = list(self.proj_in.parameters(
        )) + list(self.lin1.parameters()) + list(self.lin2.parameters())
        self.params2 = list(self.proj_out.parameters())

        self.proj_type = proj_type
        self.res = res
        self.dropout = dropout
        self.out_type = out_type

        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.proj_type == 'linear':
            nn.init.xavier_uniform_(self.proj_in.weight, gain=gain)
        if self.out_type == "direct" or self.out_type == "sum" or self.out_type == "stack":
            nn.init.xavier_uniform_(self.proj_out.weight, gain=gain)

        nn.init.xavier_uniform_(self.lin1.weight, gain=gain)
        nn.init.xavier_uniform_(self.lin2.weight, gain=gain)

    def forward(self, x, adj):
        h = x
        if self.proj_type == "linear":
            h = self.proj_in(h)
            h = F.relu(h)
        elif self.proj_type == "mlp":
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(self.lin1(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lin2(h)

        hidden_rep = [h]

        for i in range(self.layer_num):
            h = torch.mm(adj, h)
            if self.res:
                h = h + pre_h
                pre_h = h
            h = F.dropout(h, self.dropout, training=self.training)
            hidden_rep.append(h)

        if self.out_type == "stack" or self.out_type == "mlp_stack":
            score_over_layer = torch.hstack(hidden_rep)
            h = self.proj_out(score_over_layer)
        elif self.out_type == "sum" or self.out_type == "mlp_sum":
            score_over_layer = 0.
            for h in hidden_rep:
                score_over_layer += h
            h = self.proj_out(score_over_layer)
        else:
            h = self.proj_out(hidden_rep[-1])

        out = F.log_softmax(h, dim=-1)
        return out


class FCN_v6_2(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, dropout, res=False):
        super(FCN_v6_2, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer

        self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        self.lin1 = nn.Linear(nfeat, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)

        self.theta = nn.Parameter(torch.empty(nlayer+1, 1))

        # self.params1 = list(self.proj_in.parameters()) + \
        #     list(self.lin1.parameters()) + \
        #     list(self.lin2.parameters())
        # self.params2 = list(self.proj_out.parameters()) + [self.theta]

        self.res = res
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lin1.weight, gain=gain)
        nn.init.xavier_uniform_(self.lin2.weight, gain=gain)
        nn.init.xavier_uniform_(self.theta, gain=gain)

    def forward(self, x, adj):
        h = x
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        hidden_rep = [h]

        for i in range(self.layer_num):
            # h = torch.spmm(adj, h)
            h = torch.mm(adj, h)
            if self.res:
                h = h + pre_h
                pre_h = h
            h = F.dropout(h, self.dropout, training=self.training)
            hidden_rep.append(h)

        h_rep = torch.stack(hidden_rep, dim=1)
        # print(h_rep.shape)
        h_rep = torch.mul(h_rep, self.theta)
        h = torch.sum(h_rep, dim = 1)
        
        h = self.proj_out(h)
        out = F.log_softmax(h, dim=-1)
        return out


class FCN_v7_1(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, alpha, dropout,
                 proj_type='linear', out_type="direct", res=False):
        super(FCN_v7_1, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer

        if proj_type == 'linear':
            self.proj_in = nn.Linear(nfeat, mlp_hidden_dim)
        else:
            self.proj_in = MLP_v2(
                num_mlp_layers, nfeat, mlp_hidden_dim, mlp_hidden_dim, dropout)

        if out_type == "direct" or out_type == "sum":
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        elif out_type == "stack":
            self.proj_out = nn.Linear(mlp_hidden_dim * (nlayer + 1), nclass)

        elif out_type == "mlp" or out_type == "mlp_sum":
            self.proj_out = MLP(
                num_mlp_layers, mlp_hidden_dim, mlp_hidden_dim, nclass)

        elif out_type == "mlp_stack":
            self.proj_out = MLP(
                num_mlp_layers, mlp_hidden_dim * (nlayer + 1), mlp_hidden_dim, nclass)

        else:
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        self.params1 = list(self.proj_in.parameters())
        self.params2 = list(self.proj_out.parameters())

        self.proj_type = proj_type
        self.dropout = dropout
        self.out_type = out_type
        self.alpha = alpha

    def forward(self, x, adj, fe_adj):
        h = x
        h = self.proj_in(h)
        if self.proj_type == "linear":
            h = F.relu(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        pre_h = h
        hidden_rep = [h]

        for i in range(self.layer_num):
            # h1 = torch.spmm(adj, h)
            h1 = torch.mm(adj, h)
            h1 = F.dropout(h, self.dropout, training=self.training)
            # h2 = torch.spmm(fe_adj, h)
            h2 = torch.mm(fe_adj, h)
            h2 = F.dropout(h2, self.dropout, training=self.training)
            h = self.alpha * h2 + (1-self.alpha) * h1
            hidden_rep.append(h)

        if self.out_type == "stack" or self.out_type == "mlp_stack":
            score_over_layer = torch.hstack(hidden_rep)
            h = self.proj_out(score_over_layer)
        elif self.out_type == "sum" or self.out_type == "mlp_sum":
            score_over_layer = 0.
            for h in hidden_rep:
                score_over_layer += h
            h = self.proj_out(score_over_layer)
        else:
            h = self.proj_out(hidden_rep[-1])

        out = F.log_softmax(h, dim=-1)
        return out


class FCN_v8(nn.Module):
    def __init__(self, nlayer, nfeat, nclass, mlp_config, dropout,
                 proj_type='mlp', out_type="direct", res=False):
        super(FCN_v8, self).__init__()

        num_mlp_layers = mlp_config["num_mlp_layers"]
        mlp_hidden_dim = mlp_config["mlp_hidden_dim"]

        self.batch_norms = torch.nn.ModuleList()
        self.linears_prediction = torch.nn.ModuleList()

        self.layer_num = nlayer

        self.lin1 = nn.Linear(nfeat, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)

        if out_type == "direct" or out_type == "sum":
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        elif out_type == "stack":
            self.proj_out = nn.Linear(mlp_hidden_dim * (nlayer + 1), nclass)

        elif out_type == "mlp" or out_type == "mlp_sum":
            self.proj_out = MLP_v2(
                num_mlp_layers, mlp_hidden_dim, mlp_hidden_dim, nclass, dropout)

        elif out_type == "mlp_stack":
            self.proj_out = MLP_v2(
                num_mlp_layers, mlp_hidden_dim * (nlayer + 1), mlp_hidden_dim, nclass, dropout)

        else:
            self.proj_out = nn.Linear(mlp_hidden_dim, nclass)

        self.W_A = torch.nn.ModuleList()
        self.W_F = torch.nn.ModuleList()
        self.a = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.W_A.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            self.W_F.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            self.a.append(nn.Linear(2 * mlp_hidden_dim, 1))

        # self.params1 = list(self.proj_in.parameters())
        # self.params2 = list(self.proj_out.parameters())

        self.proj_type = proj_type
        self.dropout = dropout
        self.out_type = out_type

    def forward(self, x, adj, fe_adj):
        h = x
        # h = self.proj_in(h)
        # if self.proj_type == "linear":
        #     h = F.relu(h)
        # # h = F.dropout(h, self.dropout, training=self.training)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        hidden_rep = [h]

        for i in range(self.layer_num):
            # h1 = torch.spmm(adj, h)
            h1 = torch.mm(adj, h)
            h1 = F.dropout(h, self.dropout, training=self.training)
            # h2 = torch.spmm(fe_adj, h)
            h2 = torch.mm(fe_adj, h)
            h2 = F.dropout(h2, self.dropout, training=self.training)

            _h1 = self.W_A[i](h1)
            _h2 = self.W_F[i](h2)

            # aij = torch.leaky_relu(self.a[i](torch.tanh(torch.hstack([_h1, _h2]))))
            # aij = torch.tanh(self.a[i](torch.hstack([_h1, _h2])))
            aij = torch.sigmoid(self.a[i](torch.hstack([_h1, _h2])))
            # aij = torch.sigmoid(self.a[i](torch.tanh(torch.hstack([_h1, _h2]))))
            h = aij * h1 + (1 - aij) * h2
            # h = F.dropout(h, self.dropout, training=self.training)

            hidden_rep.append(h)

        if self.out_type == "stack" or self.out_type == "mlp_stack":
            score_over_layer = torch.hstack(hidden_rep)
            h = self.proj_out(score_over_layer)
        elif self.out_type == "sum" or self.out_type == "mlp_sum":
            score_over_layer = 0.
            for h in hidden_rep:
                score_over_layer += h
            h = self.proj_out(score_over_layer)
        else:
            h = self.proj_out(hidden_rep[-1])

        out = F.log_softmax(h, dim=-1)
        return out
