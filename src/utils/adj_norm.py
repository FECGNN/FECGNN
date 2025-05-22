# -*-coding:utf-8-*-
import torch
import numpy as np
import scipy.sparse as sp


# A' = D^-1/2 * A * D^-1/2
def normalized_adjacency(adj):
    # adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0)*1+row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# A' = (D)^-1/2 * ( A ) * (D )^-1/2 + I
def normalized_adjacency_selfLoop(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) + sp.eye(adj.shape[0])
    return res.tocoo()


# pytorch implementation
# A' = D^-1/2 * A * D^-1/2
def normalized_adjacency_torch(adj, device="cpu"):
    # adj = adj + torch.eye(adj.shape[0])
    row_sum = torch.sum(adj, 1)
    # row_sum = (row_sum == 0)*1+row_sum
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    support = torch.mm(d_mat_inv_sqrt, adj)
    return torch.mm(support, d_mat_inv_sqrt)


# pytorch implementation
# A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
def aug_normalized_adjacency_torch(adj, device="cpu"):
    adj = adj + torch.eye(adj.shape[0]).to(device)
    row_sum = torch.sum(adj, 1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    support = torch.mm(d_mat_inv_sqrt, adj)
    return torch.mm(support, d_mat_inv_sqrt)


# pytorch implementation
# A' = (D)^-1/2 * ( A ) * (D )^-1/2 + I
def normalized_adjacency_selfLoop_torch(adj, device="cpu"):
    row_sum = torch.sum(adj, 1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    support = torch.mm(d_mat_inv_sqrt, adj)
    support = torch.mm(support, d_mat_inv_sqrt)
    return support + torch.eye(adj.shape[0]).to(device)


# -------------------------------------------------------------------------
def fetch_normalization(type):
    switcher = {
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'GCN_1selfLoop_2normal': aug_normalized_adjacency,
        # A' = (D)^-1/2 * ( A ) * (D )^-1/2 + I
        'pow_1normal_2selfLoop': normalized_adjacency_selfLoop,
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def fetch_normalization_torch(type):
    switcher = {
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'GCN_1selfLoop_2normal': aug_normalized_adjacency_torch,
        # A' = (D)^-1/2 * ( A ) * (D )^-1/2 + I
        'pow_1normal_2selfLoop': normalized_adjacency_selfLoop_torch,
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func
# -------------------------------------------------------------------------
