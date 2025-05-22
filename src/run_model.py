import time
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.optim as optim
import torch.nn.functional as F

import models.model as model
import models.madgap as madgap
import utils.utils as utils

from utils.utils import tab_printer
from utils.adj_norm import *
from parameter_parser import parameter_parser
from models.data_loader import load_citation
from models.exp_calc import exp_calc, exp_calc_v2

global_configs = {}


def evalution(args, net,
              features,
              y_val,
              val_mask,
              configs,
              calc_mad=False):
    # Validation
    net.eval()

    t1 = time.time()

    embeds = net(features, configs["support"])

    t2 = time.time()

    val_idx = np.nonzero(val_mask)
    mask = torch.LongTensor(val_mask)
    # acc, _n = utils.masked_accuracy(embeds, y_val, mask)
    acc = utils.accuracy(embeds[val_idx], y_val[val_idx].to(args.device))

    mad_value_adj = 0.
    mad_value_full = 0.
    if calc_mad and args.madgap:
        adj = configs["adj"]
        embeds_np = embeds.cpu().detach().numpy()

        mask_arr = adj.nonzero()
        ones = np.ones(len(mask_arr[0]))
        mask_arr = sp.coo_matrix(
            (ones, (mask_arr[0], mask_arr[1]))).todense()
        mad_value_adj = madgap.mad_value(embeds_np, mask_arr)

        ones = np.ones(adj.shape[0] * adj.shape[1]
                       ).reshape(adj.shape[0], adj.shape[1])
        # ones = np.ones(len(val_idx) * len(val_idx)
        #                ).reshape(len(val_idx), len(val_idx))
        # embeds_np = embeds_np[val_idx]
        mask_arr = sp.coo_matrix(ones, dtype=np.int32).todense()

        mad_value_full = madgap.mad_value(embeds_np, mask_arr)

    if args.cuda:
        acc = acc.cpu()
    acc_val = acc.item()
    return acc_val, mad_value_adj, mad_value_full, t2 - t1


def run(args, net, optimizer, features, support,
        train_mask, val_mask, test_mask, y_train, y_val, y_test,
        configs, logger):

    val_best = 0.
    epoch_best = -1
    step_best = 0
    state_buffer = {}
    t_all = 0.
    nan_flag = False
    for epoch in range(args.epochs):
        t1 = time.time()
        net.train()
        optimizer.zero_grad()
        loss = 0.

        embeds = net(features, support)

        train_idx = np.nonzero(train_mask)
        loss = F.nll_loss(embeds[train_idx], y_train[train_idx])

        t2 = time.time()
        t_all += (t2 - t1)

        if args.cuda:
            l = loss.cpu().detach().numpy()
        else:
            l = loss.detach().numpy()

        if np.nan in l or np.isnan(l):
            logger.info(
                "epoch %04d: time cost: %.6fs, nan break. :-(" % (epoch+1, t2-t1))
            nan_flag = True
            break

        epoch_info = "epoch %04d: time cost: %.6fs, loss: %.6f, " % (
            epoch+1, t2-t1, l)

        loss.backward()
        optimizer.step()

        acc_val, mad_value_adj, mad_value_full, __ = evalution(args, net, features,
                                                               y_val,
                                                               val_mask,
                                                               configs,
                                                               calc_mad=args.val_mad)

        epoch_info += "val acc: %.6f" % (acc_val)

        if args.val_mad:
            epoch_info += ", madgap: %.4f" % (mad_value_full)

        if args.output_skip and (epoch+1) % 10 != 0:
            pass
        else:
            logger.info(epoch_info)

        if acc_val > val_best:
            val_best = acc_val
            epoch_best = epoch + 1
            state_buffer[epoch_best] = net.state_dict()
            step_best = 0
        else:
            step_best += 1

        if args.early_stop and step_best > args.patiences:
            logger.info("early stop")
            break

    if epoch_best == -1 and nan_flag:
        logger.info('NaN Break.\n')
        return -1, 0, 0, 0, 0, 0, 0

    if epoch_best == -1:
        val_best = acc_val
        epoch_best = args.epochs + 1
        state_buffer[epoch_best] = net.state_dict()

    logger.info('Finish.\n')

    state_dict_load = state_buffer[epoch_best]
    net.load_state_dict(state_dict_load)

    acc_test, mad_value_adj, mad_value_full, eval_time = evalution(args, net, features,
                                                                   y_test,
                                                                   test_mask,
                                                                   configs,
                                                                   calc_mad=True)

    ave_time = t_all / args.epochs
    return epoch_best, acc_test, ave_time, eval_time, mad_value_full, mad_value_adj, val_best


def init_configs(args):
    if args.cuda and torch.cuda.is_available():
        args.cuda = args.cuda
        args.device = torch.device("cuda")
    else:
        args.device = "cpu"
        args.cuda = False
        args.cuda_parallel = False

    if args.no_exp:
        args.exp_frac = 0.

    ct = time.time()
    time_info = time.strftime("%Y_%m_%d_%H%M%S", time.localtime(ct))
    data_secs = (ct - int(ct)) * 1000
    data_secs = "%03d" % (data_secs)
    time_info += "_" + str(data_secs)
    file_name = time_info + "_" + args.model_selec + "_" + args.dataset
    file_name += "_cuda" if args.cuda else "_cpu"
    args.file_name = file_name
    args.save_file = "./saved/" + file_name + ".state_dict.pkl"
    return args


def load_dataset(args):
    args.dataset = args.dataset.lower()
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, \
            labels = load_citation(dataset_str=args.dataset)

    args.logger.info("y_train: %d, y_val: %d, y_test: %d \n" %
                     (y_train.shape[0], y_val.shape[0], y_test.shape[0]))

    adj = adj.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj.eliminate_zeros()

    # ---------------- drop redundancy ----------------
    if args.drop_redundant:

        wnh_path = f"./processed/{args.dataset}/WNH/"
        wnh_file = f"./processed/{args.dataset}/WNH/{args.dataset}.WNH.{args.wnh_ver}.npy"
        wnh = np.load(wnh_file)

        assert len(wnh) == adj.shape[0]
        remove_nodes = []
        train_idx = np.nonzero(train_mask)
        val_idx = np.nonzero(val_mask)
        test_idx = np.nonzero(test_mask)
        train_idx = train_idx[0].tolist()
        val_idx = val_idx[0].tolist()
        test_idx = test_idx[0].tolist()
        val_test_idx = val_idx + test_idx
        train_val_test_idx = train_idx + val_test_idx
        # print(">>>>>>>>     val_test_idx:", val_test_idx)
        args.logger.info(">>>>>>>>     adj_shape: %s" % str(adj.shape))
        args.logger.info(">>>>>>>>     train_idx: %d" % len(train_idx))
        args.logger.info(">>>>>>>>     val_idx: %d" % len(val_idx))
        args.logger.info(">>>>>>>>     test_idx: %d" % len(test_idx))
        args.logger.info(">>>>>>>>     val_test_idx: %d" % len(val_test_idx))
        
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            other_idx = [i for i in list(
                range(adj.shape[0])) if i not in train_val_test_idx]
            args.logger.info(">>>>>>>>     other_idx: %d" % len(other_idx))

        G = nx.from_numpy_matrix(adj)
        # print(args.dataset, "# node:", G.number_of_nodes(),
        #       ", # edge:", G.number_of_edges())

        n_list = [(n, G.degree(n)) for n in G.nodes()]
        n_list.sort(key=lambda x: x[1], reverse=True)

        th_degree = args.th_degree
        th_wnh = args.th_wnh

        p_drop_rate = args.p_drop_rate
        wnh_list = [(n, wnh[n]) for n in G.nodes()]
        wnh_list.sort(key=lambda x: x[1], reverse=True)
        p_idx = int(p_drop_rate * len(wnh_list))
        remove_candidates = [n[0] for n in wnh_list[:p_idx]]

        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            for _n, _degree in n_list:
                if _n not in test_idx and _degree > th_degree and _n in remove_candidates:
                    remove_nodes.append(_n)

        global_configs["remove_nodes"] = remove_nodes

        adj = adj.todense()
        adj = np.delete(adj, remove_nodes, 0)
        adj = np.delete(adj, remove_nodes, 1)

        args.logger.info("")
        args.logger.info("! >>>>>>     remove_candidates: %d" % len(remove_candidates))
        args.logger.info("! >>>>>>     remove_nodes: %d" % len(remove_nodes))
        args.logger.info(">>>>>>>>     adj_shape: %s" % str(adj.shape))

        adj = sp.csr_matrix(adj)

        features = features.tolil()
        features.rows = np.delete(features.rows, remove_nodes)
        features.data = np.delete(features.data, remove_nodes)
        features._shape = (
            features._shape[0] - len(remove_nodes), features._shape[1])

        y_train = np.delete(y_train, remove_nodes, 0)
        y_val = np.delete(y_val, remove_nodes, 0)
        y_test = np.delete(y_test, remove_nodes, 0)
        train_mask = np.delete(train_mask, remove_nodes, 0)
        val_mask = np.delete(val_mask, remove_nodes, 0)
        test_mask = np.delete(test_mask, remove_nodes, 0)
        labels = np.delete(labels, remove_nodes, 0)

        args.logger.info("")
        args.logger.info(">>>>>>>>     y_train: %s" % str(y_train.shape))
        args.logger.info(">>>>>>>>     y_val: %s" % str(y_val.shape))
        args.logger.info(">>>>>>>>     y_test: %s" % str(y_test.shape))

        args.logger.info(">>>>>>>>     train_mask: %s" % str(train_mask.shape))
        args.logger.info(">>>>>>>>     val_mask: %s" % str(val_mask.shape))
        args.logger.info(">>>>>>>>     test_mask: %s" % str(test_mask.shape))
        args.logger.info(">>>>>>>>     labels: %s" % str(labels.shape))
        args.logger.info("")

    # ---------------- drop redundancy end ------------

    global_configs["train_mask"] = train_mask
    global_configs["val_mask"] = val_mask
    global_configs["test_mask"] = test_mask

    _f = utils.sparse_mx_to_torch_sparse_tensor(features)
    global_configs["features"] = _f.to_dense()
    global_configs["y_train"] = torch.LongTensor(y_train.reshape((-1)))
    global_configs["y_val"] = torch.LongTensor(y_val.reshape((-1)))
    global_configs["y_test"] = torch.LongTensor(y_test.reshape((-1)))

    if args.cuda and torch.cuda.is_available():
        global_configs["features"] = global_configs["features"].to(args.device)
        global_configs["y_train"] = global_configs["y_train"].to(args.device)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def data_processing(args, adj, features, labels, y_train, y_val, y_test, ):
    if args.hidden_fixed > 0:
        layers_config = [features.shape[1]] + \
            [args.hidden_fixed for _ in range(args.num_layer - 1)] +\
            [labels.max().item() + 1]
        assert args.num_layer == len(layers_config) - 1
    else:
        layers_config = [features.shape[1]] + \
            list(args.hidden) +\
            [labels.max().item() + 1]
    print("layer info:", layers_config)
    print("layers:", len(layers_config) - 1)
    args.nLayers = len(layers_config) - 1
    global_configs["layers_config"] = layers_config
    n_classes = global_configs["layers_config"][-1]

    # ---------------- exp function selection ----------------
    if args.exp_func == 1:
        exp_func = exp_calc
    elif args.exp_func == 2:
        exp_func = exp_calc_v2  # including negative eigenvalue
    else:
        raise NotImplementedError
    # ---------------- exp function selection end ------------

    adj_og = adj
    global_configs["adj"] = adj_og

    adj_norm = aug_normalized_adjacency(adj)
    adj_norm = utils.sparse_mx_to_torch_sparse_tensor(adj_norm)
    adj_norm = adj_norm.to_dense()
    adj_norm = adj_norm.to(args.device)
    global_configs["adj_norm"] = adj_norm

    if args.no_exp or args.exp_frac == 1.0:
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        support = aug_normalized_adjacency(adj)
        support = utils.sparse_mx_to_torch_sparse_tensor(support)
        support = support.to_dense()
        support = support.to(args.device)

    elif args.norm_type == "gcn":
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        # A' = FE(A')
        adj_norm = aug_normalized_adjacency(adj)
        adj_norm = utils.sparse_mx_to_torch_sparse_tensor(adj_norm)
        adj_norm = adj_norm.to_dense()
        t1 = time.perf_counter()

        # support, top_e, top_v = fe_preprocessing(
        #     adj_norm, args.exp_frac, args.fVer + "_gcn", args.device, args)

        support, top_e, top_v = exp_func(
            dataset=args.dataset,
            adj=adj_norm, exp_frac=args.exp_frac,
            norm_type=args.fVer + "_gcn",
            device=args.device,
            decompose=args.force_decompose,
            saveFile=not args.no_frac_save)

        t2 = time.perf_counter()
        args.logger.info('eigen-decomposition: %.4f' % (t2 - t1))



    global_configs["support"] = support


    return 0


def build_model(args, skip_features=False):
    # initialize model
    if args.model_selec == "FCN":
        net = model.FCN(layers_config=global_configs["layers_config"],
                        dropout=args.dropout)

    elif args.model_selec in ["FCN_MLP_6", "FCN_v6", "v6"]:
        mlp_config = {"mlp_hidden_dim": args.mlp_hidden_dim,
                      "num_mlp_layers": args.num_mlp_layers}
        net = model.FCN_v6(nlayer=args.num_layer, nfeat=global_configs["layers_config"][0], nclass=global_configs["layers_config"][-1],
                           mlp_config=mlp_config, dropout=args.dropout, proj_type='linear', out_type=args.fcn_out_type)


    elif args.model_selec in ["v6_MLP", "v6mlp"]:
        mlp_config = {"mlp_hidden_dim": args.mlp_hidden_dim,
                      "num_mlp_layers": args.num_mlp_layers}
        net = model.FCN_v6(nlayer=args.num_layer, nfeat=global_configs["layers_config"][0], nclass=global_configs["layers_config"][-1],
                           mlp_config=mlp_config, dropout=args.dropout, proj_type='mlp', out_type=args.fcn_out_type)

    else:
        pass

    if args.model_selec in ["FCN_v2", "FCN_v6", "v6", "v6_MLP", "v6mlp",]:
        optimizer = optim.Adam([
            {'params': net.params1, 'weight_decay': args.wd},
            {'params': net.params2, 'weight_decay': args.wd2}, ],
            lr=args.lr,)
    else:
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               weight_decay=args.wd)

    net = net.to(args.device)
    return net, optimizer


def main():
    args = parameter_parser()
    torch.manual_seed(args.seed)
    utils.init_dir()

    args = init_configs(args)

    logger = utils.get_log("log/" + args.file_name + ".log", args.log_file)
    tab_printer(args, logger)
    args.logger = logger

    adj, features, y_train, y_val, y_test, \
        train_mask, val_mask, test_mask, labels = load_dataset(args)

    data_processing(args, adj, features, labels, y_train, y_val, y_test)

    # return

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        net, optimizer = build_model(args)
        
        epoch_best, acc_test, ave_time, eval_time, mad_value_full, mad_value_adj, val_best = run(
            args, net, optimizer, global_configs["features"], global_configs["support"],
            global_configs["train_mask"], global_configs["val_mask"], global_configs["test_mask"],
            global_configs["y_train"], global_configs["y_val"], global_configs["y_test"],
            global_configs, logger)

        if epoch_best == -1:
            # nan break
            return

        logger.info("layers: %d\n" % (args.nLayers))
        logger.info("average time: %.3f\n" % (ave_time))
        logger.info("eval time: %.3f\n" % (eval_time))
        logger.info("best epoch: %04d\n" % (epoch_best))
        logger.info("val acc: %.6f\n" % (val_best))
        logger.info("test acc: %.6f\n" % (acc_test))

        if args.madgap:
            logger.info("test mad full: %.6f\n" % (mad_value_full))
            logger.info("test mad adj: %.6f\n" % (mad_value_adj))

            logger.info("%04d %.6f %.6f %.6f %.3f %.3f\n" % (
                epoch_best, acc_test, mad_value_full, mad_value_adj, ave_time, eval_time))
        else:
            logger.info("%04d %.6f %.3f %.3f\n" % (
                epoch_best, acc_test, ave_time, eval_time))

    args.logger = None
    if args.end_log:
        tab_printer(args, logger, is_end=True)


if __name__ == "__main__":
    main()
