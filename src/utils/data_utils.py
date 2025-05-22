import random
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


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


def sample_mask_to_idx(mask):
    x_idx = np.where(mask == 1)
    x_idx = np.array(x_idx[0], dtype=np.int32)
    return x_idx


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


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.astype(float)
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0)*1+row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    """Standardize features by removing the mean and scaling to unit variance
       The standard score of a sample `x` is calculated as:    z = (x - u) / s """
    scaler = StandardScaler()
    mx = scaler.fit_transform(mx)
    return mx
