# -*-coding:utf-8-*-
from collections import defaultdict
import logging
import random
import os

import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

from texttable import Texttable
from scipy.sparse import coo_matrix

from utils.adj_norm import fetch_normalization


def tab_printer(args, logger, is_end=False):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +
               [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    if is_end:
        logger.info("\n\nLr:\t" + format(args["lr"], "f") + "\n")
        logger.info("\n" + t.draw() + "\n\n\n\n\n")
    else:
        logger.info("\n" + t.draw())
        logger.info("\n\nLr:\t" + format(args["lr"], "f") + "\n")


# def graph_reader(path):
#     """
#     Function to read the graph from the path.
#     :param path: Path to the edge list.
#     :return graph: NetworkX object returned.
#     """
#     graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
#     return graph


# def feature_reader(path):
#     """
#     Reading the sparse feature matrix stored as csv from the disk.
#     :param path: Path to the csv file.
#     :return features: Dense matrix of features.
#     """
#     features = pd.read_csv(path)
#     node_index = features["node_id"].values.tolist()
#     feature_index = features["feature_id"].values.tolist()
#     feature_values = features["value"].values.tolist()
#     node_count = max(node_index)+1
#     feature_count = max(feature_index)+1
#     features = coo_matrix((feature_values, (node_index, feature_index)), shape=(
#         node_count, feature_count)).toarray()
#     return features


# def target_reader(path):
#     """
#     Reading the target vector from disk.
#     :param path: Path to the target.
#     :return target: Target vector.
#     """
#     target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
#     print("tartget shape:", target.shape)
#     return target


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def _accuracy(preds, labels):
    # preds = output.max(1)[1].type_as(labels)
    correct = np.sum(preds == labels)
    return correct / len(labels)


def masked_accuracy(output, labels, mask):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.mul(mask)
    correct = correct.sum()
    n = mask.sum()
    return correct / n, n


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    m_indices = torch_sparse._indices().numpy()
    row, col = m_indices[0], m_indices[1]
    data = torch_sparse._values().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)),
                              shape=(torch_sparse.size()[0],
                                     torch_sparse.size()[1]))
    return sp_matrix


def get_random_features_lil_matrix(num_row, num_cul, seed):
    # 对每一行有元素的个数先随机出来，之后再对每行进行随机
    random.seed(seed)
    np.random.seed(seed)
    full_row = range(num_cul)
    fin_x = None
    fin_y = None
    for i in range(num_row):
        t_nonzero = random.randint(0, num_cul)  # 0 <= N <= num
        t_y = random.choices(full_row, k=t_nonzero)
        t_x = [i for j in range(t_nonzero)]
        if fin_x is None:
            fin_x = t_x
        else:
            fin_x = fin_x + t_x
        if fin_y is None:
            fin_y = t_y
        else:
            fin_y = fin_y + t_y

    fin_d = [1 for j in range(len(fin_x))]
    res_sparse = sp.csr_matrix((fin_d, (fin_x, fin_y)), shape=(
        num_row, num_cul), dtype='float').tolil()

    return res_sparse


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_adj_list(adj_matrix):
    """build adjacency list from adjacency matrix"""
    adj_list = {}
    for i in range(adj_matrix.shape[0]):
        adj_list[i] = set(np.where(adj_matrix[i].toarray() != 0)[1])
    return adj_list


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="GCN_1selfLoop_2normal"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def get_log(file_name, log_file=False):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.DEBUG)  # 设定输出hander的level
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    logger.addHandler(ch)  # 将两个hander添加到我们声明的logger中去

    if log_file:
        # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
        fh = logging.FileHandler(file_name, mode='a')
        fh.setLevel(logging.DEBUG)  # 设定文件hander得lever
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def to_adj_dict(csr_adj):
    adj = csr_adj.toarray()
    ind = np.nonzero(adj)
    adj_lists = defaultdict(set)
    for u, v in zip(ind[0], ind[1]):
        adj_lists[u].add(v)
        adj_lists[v].add(u)
    return adj_lists


def init_dir():
    datasets = ['cora', 'citeseer', 'pubmed',
                'chameleon', 'squirrel', 'film',
                'computers', 'photo']
    processed_path = "./processed/"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    for d in datasets:
        d = d.lower()
        d_path = processed_path + d + "/"
        if not os.path.exists(d_path):
            os.makedirs(d_path)
        d_wnh_path = processed_path + d + "/WNH/"
        if not os.path.exists(d_wnh_path):
            os.makedirs(d_wnh_path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
