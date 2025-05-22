import os
import random
import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx
from pathlib import Path
import pickle

from utils.data_utils import *


def load_citation(dataset_str="cora", seed=42, use_random_feature=False):
    """
    Load Citation Networks Datasets.
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    data_source_dir = f"./data/{dataset_str.lower()}/"
    for i in range(len(names)):
        with open(data_source_dir + "ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        data_source_dir + "ind.{}.test.index".format(dataset_str))  # ../data/
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = row_normalize(features)

    if use_random_feature:
        random_feature_file = Path(data_source_dir + "ind.{}.{}".format(
            dataset_str.lower(), 'random_all_features') + '_cpu.pickle')
        if random_feature_file.exists():
            with open(data_source_dir + "ind.{}.{}".format(dataset_str.lower(), 'random_all_features') + '_cpu.pickle', 'rb') as f:
                features = pickle.load(f)
                f.close()
            print('Existed random featrues have been road')
        else:
            features = get_random_features_lil_matrix(
                features.shape[0], features.shape[1], seed)
            with open(data_source_dir + "ind.{}.{}".format(dataset_str.lower(), 'random_all_features') + '_cpu.pickle', 'wb') as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
            print('Random featrues have been computed and saved')

    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])

    print(dataset_str, "# node:", G.number_of_nodes(),
          ", # edge:", G.number_of_edges())

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1).reshape((-1, 1))

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    edges = []
    for nd_idx in graph.keys():
        for nb_idx in graph[nd_idx]:
            edges.append((nd_idx, nb_idx))
    num_data = len(list(graph.keys()))
    print(num_data, features.shape, adj.shape)

    test_data = test_idx_range
    val_data = np.array(idx_val, dtype=np.int32)
    train_data = np.array(idx_train, dtype=np.int32)

    # visible data for cluster_gcn training
    is_visible = np.ones((num_data), dtype=np.bool)
    is_visible[val_data] = False
    is_visible[test_data] = False
    visible_data = np.array([n for n in range(num_data) if is_visible[n]],
                            dtype=np.int32)

    visible_edges = [
        (e[0], e[1]) for e in edges if is_visible[e[0]] and is_visible[e[1]]
    ]
    edges = np.array(edges, dtype=np.int32)
    visible_edges = np.array(visible_edges, dtype=np.int32)

    def _construct_adj(edges):
        adj = sp.csr_matrix((np.ones(
            (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(num_data, num_data))
        adj += adj.transpose()
        return adj

    train_adj = _construct_adj(visible_edges)
    all_adj = _construct_adj(edges)

    train_feats = features[visible_data]
    test_feats = features

    print("features:", features.shape, train_feats.shape, visible_data.shape)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]

    train_mask = sample_mask(train_data, labels.shape[0])
    val_mask = sample_mask(val_data, labels.shape[0])
    test_mask = sample_mask(test_data, labels.shape[0])

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


""" 
    full_supervised_datasets: 
"""
def full_load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, _1, _2, _3, _4, _5, _6, labels = load_citation(dataset_name)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join(
            './data/geom_gcn_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('./data/geom_gcn_data', dataset_name,
                                                                'out1_node_feature_label.txt')
        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(
                    line[0]) not in graph_labels_dict)
                if dataset_name in ['film', ]:
                    feature_amount = 932
                    _feats = np.zeros(feature_amount, dtype=np.uint8)
                    _idx = [int(_i) for _i in line[1].split(',')]
                    _feats[_idx] = 1
                    graph_node_features_dict[int(line[0])] = _feats
                else:
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

            features = np.array(
                [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            labels = np.array(
                [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

        print(dataset_name, "# node:", G.number_of_nodes(),
              ", # edge:", G.number_of_edges())

        features = sp.coo_matrix(features)
        features = row_normalize(features)

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    g = adj

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    # features = torch.FloatTensor(features)
    # labels = torch.LongTensor(labels)
    # train_mask = torch.BoolTensor(train_mask)
    # val_mask = torch.BoolTensor(val_mask)
    # test_mask = torch.BoolTensor(test_mask)

    # g = sys_normalized_adjacency(g)
    # g = sparse_mx_to_torch_sparse_tensor(g)

    train_data = sample_mask_to_idx(train_mask)
    val_data = sample_mask_to_idx(val_mask)
    test_data = sample_mask_to_idx(test_mask)

    labels = labels.reshape((-1, 1))

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]

    # return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


""" 
    large datasets: 
"""


def load_large_graph(dataset_name, split_seed=42, row_norm=False):
    if dataset_name in ["mag_scholar_c"]:
        data_path = "data/graph_large/{}/{}".format(dataset_name, dataset_name)
        data_set = np.load(data_path + '.npz')
        adj_data = data_set['adj_matrix.data']
        adj_indices = data_set['adj_matrix.indices']
        adj_indptr = data_set['adj_matrix.indptr']
        adj_shape = data_set['adj_matrix.shape']

        feat_data = data_set['attr_matrix.data']
        feat_indices = data_set['attr_matrix.indices']
        feat_indptr = data_set['attr_matrix.indptr']
        feat_shape = data_set['attr_matrix.shape']
        labels_num = data_set['labels']
        features = sp.csr_matrix(
            (feat_data, feat_indices, feat_indptr), shape=feat_shape)
        adj = sp.csr_matrix(
            (adj_data, adj_indices, adj_indptr), shape=adj_shape)
        label_count = labels_num.max() + 1
        labels = np.eye(label_count)[labels_num]
        if row_norm:
            features = features.tocoo()
            features = row_normalize(features)
    else:
        data_path = "data/graph_large/{}/{}".format(dataset_name, dataset_name)
        adj = sp.load_npz(data_path + "_adj.npz")
        features = np.load(data_path + "_feat.npy")
        labels = np.load(data_path + "_labels.npy")
        if row_norm:
            features = features.tocoo()
            features = row_normalize(features)
        elif dataset_name in ['aminer']:
            features = col_normalize(features)

    print(dataset_name)
    print(adj.shape)
    print(features.shape)
    print(labels.shape)
    print()

    # random_state = np.random.RandomState(split_seed)
    # if dataset_name in ['mag_scholar_c']:
    #     idx_train, idx_val, idx_test = get_train_val_test_split(
    #         random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
    #     idx_unlabel = np.concatenate((idx_val, idx_test))
    # elif dataset_name in ['reddit']:
    #     idx_train, idx_val, idx_test = get_train_val_test_split(
    #         random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
    #     idx_unlabel = np.concatenate((idx_val, idx_test))
    # elif dataset_name in ['Amazon2M']:
    #     class_num = labels.shape[1]
    #     idx_train, idx_val, idx_test = get_train_val_test_split(
    #         random_state, labels, train_size=20 * class_num, val_size=30 * class_num)
    #     idx_unlabel = np.concatenate((idx_val, idx_test))
    # elif dataset_name in ['aminer']:
    #     idx_train, idx_val, idx_test = get_train_val_test_split(
    #         random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
    #     idx_unlabel = np.concatenate((idx_val, idx_test))
    #     features = col_normalize(features)

    # labels = np.argmax(labels, axis=1).reshape((-1, 1))

    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        get_large_idx(dataset_name, labels, split_seed)

    features = sp.coo_matrix(features)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def get_large_idx(dataset_name, labels, split_seed=42):
    random_state = np.random.RandomState(split_seed)
    if dataset_name in ['mag_scholar_c']:
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    elif dataset_name in ['reddit']:
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    elif dataset_name in ['Amazon2M']:
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_size=20 * class_num, val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    elif dataset_name in ['aminer']:
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))

    _labels = np.argmax(labels, axis=1).reshape((-1, 1))
    train_mask = sample_mask(idx_train, _labels.shape[0])
    val_mask = sample_mask(idx_val, _labels.shape[0])
    test_mask = sample_mask(idx_test, _labels.shape[0])

    y_train = np.zeros(_labels.shape)
    y_val = np.zeros(_labels.shape)
    y_test = np.zeros(_labels.shape)
    y_train[train_mask, :] = _labels[train_mask, :]
    y_val[val_mask, :] = _labels[val_mask, :]
    y_test[test_mask, :] = _labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask,


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
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
