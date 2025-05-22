import numpy as np
import scipy.sparse as sp
import networkx as nx

"""https://github.com/shchur/gnn-benchmark/"""


def _row_normalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.astype(float)
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def _eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def _is_directed(adj_matrix):
    """Check if the graph is directed (adjacency matrix is not symmetric)."""
    return (adj_matrix != adj_matrix.T).sum() != 0


def _is_undirected(adj_matrix):
    """Check if the graph is undirected (adjacency matrix is not symmetric)."""
    return (adj_matrix != adj_matrix.T).sum() == 0


def _is_weighted(adj_matrix):
    """Check if the graph is weighted (edge weights other than 1)."""
    return np.any(np.unique(adj_matrix[adj_matrix != 0].A1) != 1)


def _to_unweighted(adj_matrix):
    """Convert to an unweighted graph (set all edge weights to 1)."""
    adj_matrix.data = np.ones_like(adj_matrix.data)
    return adj_matrix


def _to_undirected(adj_matrix):
    """Convert to an undirected graph (make adjacency matrix symmetric)."""
    if _is_weighted(adj_matrix):
        raise ValueError("Convert to unweighted graph first.")
    else:
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix[adj_matrix != 0] = 1
        return adj_matrix


def read_npz(path):
    with np.load(path) as f:
        return parse_npz(f)


def parse_npz(f):
    x = sp.csr_matrix((f['attr_data'], f['attr_indices'], f['attr_indptr']),
                      f['attr_shape'])
    x[x > 0] = 1
    x = _row_normalize(x)

    adj = sp.csr_matrix((f['adj_data'], f['adj_indices'], f['adj_indptr']),
                        f['adj_shape'])
    adj = _eliminate_self_loops(adj)
    adj = _to_undirected(adj)

    labels = np.array(f['labels'])
    # labels = np.argmax(labels, axis=1).reshape((-1, 1))
    return x, adj, labels


def get_amazon_dataset(dataset_str):
    path = f"./data/{dataset_str}/amazon_electronics_{dataset_str}.npz"
    features, adj, labels = read_npz(path)
    # labels = labels.reshape((-1, 1))
    labels = _encode_onehot(labels)

    _adj = adj.todense()
    G = nx.from_numpy_matrix(_adj)
    print(dataset_str, "# node:", G.number_of_nodes(),
          ", # edge:", G.number_of_edges())
    print("is_undirect:", _is_undirected(adj))

    train_mask, val_mask, test_mask, y_train, y_val, y_test = get_split(
        42, labels)

    # return features, adj, labels,
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def get_split(split_seed, labels, forbidden_nodes=None):
    random_state = np.random.RandomState(split_seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, train_examples_per_class=20, val_examples_per_class=30,
        forbidden_nodes=forbidden_nodes)

    # print(idx_train.shape)
    # print(idx_val.shape)
    # print(idx_test.shape)

    # print(idx_train[:10])
    # print(idx_val[:10])
    # print(idx_test[:10])
    # print()

    # num_samples, num_classes = labels.shape
    # # get indices sorted by class
    # for class_index in range(num_classes):
    #     count = 0
    #     for sample_index in idx_train:
    #         if labels[sample_index, class_index] > 0.0:
    #             count += 1
    #     print('[train] class:', class_index, count)
    # print()

    # for class_index in range(num_classes):
    #     count = 0
    #     for sample_index in range(num_samples):
    #         if labels[sample_index, class_index] > 0.0:
    #             count += 1
    #     print('[all] class:', class_index, count)
    # print()

    _labels = np.argmax(labels, axis=1).reshape((-1, 1))
    train_mask = _sample_mask(idx_train, _labels.shape[0])
    val_mask = _sample_mask(idx_val, _labels.shape[0])
    test_mask = _sample_mask(idx_test, _labels.shape[0])

    y_train = np.zeros(_labels.shape)
    y_val = np.zeros(_labels.shape)
    y_test = np.zeros(_labels.shape)
    y_train[train_mask, :] = _labels[train_mask, :]
    y_val[val_mask, :] = _labels[val_mask, :]
    y_test[test_mask, :] = _labels[test_mask, :]

    print(y_train.shape)

    return train_mask, val_mask, test_mask, y_train, y_val, y_test


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None,
                             forbidden_nodes=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    forbidden_indices = []
    if forbidden_nodes is not None:
        print(len(forbidden_nodes))
        remaining_indices = [
            i for i in remaining_indices if i not in forbidden_nodes]
        # remaining_indices = remaining_indices - forbidden_nodes
        forbidden_indices = forbidden_nodes

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class, forbidden_indices = forbidden_indices)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    forbidden_indices = np.concatenate((forbidden_indices, train_indices))
    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=forbidden_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((forbidden_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        if forbidden_nodes is not None:
            # all indices must be part of the split
            # print(len(train_indices))
            # print(len(val_indices))
            # print(len(test_indices))
            # print(len(np.concatenate((train_indices, val_indices, test_indices))))
            # print(num_samples)
            # print(len(forbidden_nodes))
            # print(num_samples-len(forbidden_nodes))
            assert len(np.concatenate(
                (train_indices, val_indices, test_indices))) == num_samples-len(forbidden_nodes)
        else:
            # all indices must be part of the split
            assert len(np.concatenate(
                (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
