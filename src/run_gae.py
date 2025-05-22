import os
import time
import argparse

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

from models.data_loader import load_citation, full_load_data, load_large_graph, get_large_idx
from models.data_loader_npz import get_amazon_dataset, get_split

from models.GVAE import GCNModelVAE
from utils.gae_utils import *
from utils.data_utils import encode_onehot


def run_graph_vae(adj, features, configs=None):
    if configs is not None:
        pass

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((
        adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()  # remove self-loop

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false \
        = mask_test_edges(adj)
    adj = adj_train

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / \
        float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                        torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                         torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                        torch.FloatTensor(features[1]),
                                        torch.Size(features[2]))

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer)
    model = GCNModelVAE(num_feats=num_features, num_nodes=num_nodes,
                        hidden1=args.hidden1, hidden2=args.hidden2,
                        dropout=args.dropout)
    optimizer = Adam(model.parameters(), lr=args.lr)

    def get_scores(edges_pos, edges_neg, adj_rec):
        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(adj_rec[e[0], e[1]].item())
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]].data)
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # train model
    for epoch in range(args.epochs):
        t = time.time()

        A_pred = model(features, adj_norm)
        optimizer.zero_grad()
        loss = log_lik = norm * \
            F.binary_cross_entropy(
                A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)

        kl_divergence = 0.5 / A_pred.size(0) * (
            1 + 2*model.log_std - model.z_mean**2 - torch.exp(model.log_std)**2).sum(1).mean()
        loss -= kl_divergence

        loss.backward()
        optimizer.step()

        train_acc = get_acc(A_pred, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "train_acc=", "{:.5f}".format(
                  train_acc), "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))

    adj_orig_norm = preprocess_graph(adj_orig)
    adj_orig_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_orig_norm[0].T),
                                             torch.FloatTensor(
                                                 adj_orig_norm[1]),
                                             torch.Size(adj_orig_norm[2]))

    print("-" * 10)
    model.eval()
    A_pred = model(features, adj_orig_norm)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))

    A_pred = A_pred.cpu().detach().numpy()
    return A_pred


def calc_dist(onehot_lables):
    diff = onehot_lables[:, None, :] - onehot_lables
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist


def calc_wnh(args, ):

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
            labels = load_citation(dataset_str=args.dataset)

    elif args.dataset in ['chameleon', 'cornell', 'texas', 'film']:
        splitstr = './data/splits/'+args.dataset + \
            '_split_0.6_0.2_'+str(0)+'.npz'
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
            labels = full_load_data(args.dataset, splits_file_path=splitstr)

    elif args.dataset in ['Amazon2M', 'aminer', 'reddit', 'mag_scholar_c']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
            labels = load_large_graph(args.dataset, split_seed=42)
        labels = np.argmax(labels, axis=1).reshape((-1, 1))

    elif args.dataset in ['computers', 'photo']:
        adj, features, y_train, y_val, y_test, \
            train_mask, val_mask, test_mask, labels = \
            get_amazon_dataset(args.dataset)
        labels = np.argmax(labels, axis=1).reshape((-1, 1))

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((
        adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    adj_orig = adj_orig.todense()

    row_sum = np.sum(adj_orig, 1)  # node degree
    row_sum = np.array((row_sum == 0) * 1 + row_sum).reshape(-1)
    # print("shape of row_sum", row_sum.shape)

    A_pred = run_graph_vae(adj, features)
    # print(A_pred)
    # print(A_pred.shape)

    _labels = labels.reshape(-1)
    onehot_labels = np.zeros((_labels.size, _labels.max() + 1))
    onehot_labels[np.arange(_labels.size), _labels] = 1

    dist = calc_dist(onehot_labels)
    adj_mask = np.zeros_like(adj_orig)
    adj_mask = np.array(adj_mask)  # to make * operation works correctly
    adj_mask[adj_orig > 0] = 1
    dist = adj_mask * dist

    w_dist = dist * A_pred
    wnh = np.sum(w_dist, -1)
    # print(wnh)
    # print("shape of wnh", wnh.shape)

    wnh = wnh / row_sum

    # print("wnh:")
    # print(wnh)
    # print()

    # print("sorted:")
    # w = np.sort(wnh, )
    # w = w[::-1]
    # l = int(len(w) * 0.2)
    # print(w[:l])

    return wnh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()
    args.dataset = 'film'
    args.hidden1 = 32
    args.hidden2 = 16
    args.lr = 0.05
    args.epochs = 600
    args.dropout = 0
    args.wnh_version = '2310v4'

    print("dataset: %s" % args.dataset)
    wnh_path = f"./processed/{args.dataset}/WNH/"
    if not os.path.exists(wnh_path):
        os.makedirs(wnh_path)
    wnh_file = f"./processed/{args.dataset}/WNH/{args.dataset}.WNH.{args.wnh_version}.npy"
    print("file path: %s" % wnh_file)

    wnh = calc_wnh(args)

    # wnh_path = Path(wnh_file)
    np.save(wnh_file, wnh)

    print(wnh[:10])
    print(wnh[10:20])
    print(wnh[20:30])
    print(wnh[30:40])

    # adj_orig = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    #                      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    #                      ])
    # print(adj_orig.shape)

    # _labels = labels.reshape(-1)[:10]
    # print(1)
    # print(adj_orig)
    # print()

    # onehot_labels = np.zeros((_labels.size, _labels.max() + 1))
    # onehot_labels[np.arange(_labels.size), _labels] = 1
    # print(2)
    # print(onehot_labels)
    # print()

    # dist = calc_dist(onehot_labels)

    # print(3)
    # print(dist)
    # print(dist.shape)
    # print()
    # print()

    # adj_mask = np.zeros_like(adj_orig)
    # adj_mask = np.array(adj_mask)
    # adj_mask[adj_orig > 0] = 1

    # # print(4)
    # # print(adj_mask)
    # # print(adj_mask.shape)
    # # print()

    # dist = adj_mask * dist
    # print(5)
    # print(dist)

    # A_10 = A_pred[:10, :10]
    # print(A_10)

    # print(6)
    # a_mask = adj_mask * A_10
    # print(a_mask)
    # print()

    # w_dist = dist * A_10
    # print(w_dist)
    # w = np.sum(w_dist, -1)
    # print(w)
