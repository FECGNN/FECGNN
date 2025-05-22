import pickle
from pathlib import Path
import torch


def exp_calc(dataset, adj, exp_frac, norm_type='A', device="cpu", decompose=False, saveFile=True):

    path_top_e = f"./processed/{dataset}/{dataset}_{norm_type}_top_e.pkl"
    path_top_v = f"./processed/{dataset}/{dataset}_{norm_type}_top_v.pkl"
    e_file, v_file = Path(path_top_e), Path(path_top_v)

    support = 0.
    aa = adj.cpu()

    if (not e_file.exists() and not v_file.exists()) or decompose:
        top_e, top_v = _symeig(aa)
        if saveFile:
            with open(path_top_e, 'wb') as f_e:
                pickle.dump(top_e, f_e, protocol=4)
            with open(path_top_v, 'wb') as f_v:
                pickle.dump(top_v, f_v, protocol=4)
    else:
        with open(path_top_e, 'rb') as f_e:
            top_e = pickle.load(f_e)
        with open(path_top_v, 'rb') as f_v:
            top_v = pickle.load(f_v)

    # print(top_e)

    support_file_path = f"./processed/{dataset}/{dataset}_{norm_type}_exp_{exp_frac}.pkl"
    support_file = Path(support_file_path)
    if (not support_file.exists()) or decompose:
        top_e_exp = top_e.pow(exp_frac)
        mid_support = torch.mm(top_v,
                               torch.diag_embed(top_e_exp))
        fin_support = torch.mm(mid_support, top_v.t())
        if saveFile:
            with open(support_file_path, 'wb') as f_support:
                pickle.dump(fin_support, f_support, protocol=4)
    else:
        with open(support_file_path, 'rb') as f_support:
            fin_support = pickle.load(f_support)

    support = fin_support.to(device)

    # print("len adj_list:", len(support_list))
    return support, top_e, top_v


def _symeig(adj):
    # The eigenvalues are returned in ascending order!
    e, v = torch.linalg.eigh(adj, "U")
    # e, v = torch.linalg.eigh(adj)
    e = _flip(e, 0)  # 利用镜像对称函数将所得升序特征值改为降序
    v = _flip(v, 1)  # 特征向量是按列存放的

    # 非负截断
    NUM_top_eig = 0  # e.shape[0]
    for i in range(e.shape[0]):
        if e[i] <= 0:
            break
        NUM_top_eig += 1

    # MODEL
    # NUM_top_eig = min(NUM_top_eig, int(e.shape[0]/2))

    top_e = e[:NUM_top_eig]
    top_v = v[:, :NUM_top_eig]
    # print('reserve/total: %d/%d' % (NUM_top_eig, e.shape[0]))
    return top_e, top_v


# 分数幂常用的辅助函数
def _flip(x, dim):  # 镜像对称
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def exp_calc_v2(dataset, adj, exp_frac, norm_type='A', device="cpu", decompose=False, saveFile=True):
    path_top_e = f"./processed/{dataset}/{dataset}_{norm_type}_top_e_v2.pkl"
    path_top_v = f"./processed/{dataset}/{dataset}_{norm_type}_top_v_v2.pkl"
    e_file, v_file = Path(path_top_e), Path(path_top_v)

    support = 0.
    aa = adj.cpu()

    if (not e_file.exists() and not v_file.exists()) or decompose:
        top_e, top_v = _symeig_v2(aa)
        if saveFile:
            with open(path_top_e, 'wb') as f_e:
                pickle.dump(top_e, f_e, protocol=4)
            with open(path_top_v, 'wb') as f_v:
                pickle.dump(top_v, f_v, protocol=4)
    else:
        with open(path_top_e, 'rb') as f_e:
            top_e = pickle.load(f_e)
        with open(path_top_v, 'rb') as f_v:
            top_v = pickle.load(f_v)

    # print(top_e)

    support_file_path = f"./processed/{dataset}/{dataset}_{norm_type}_exp_{exp_frac}_v2.pkl"
    support_file = Path(support_file_path)
    if (not support_file.exists()) or decompose:

        mask_top_e = torch.ones_like(top_e)
        mask_top_e[top_e < 0] = -1
        top_e = top_e.abs()
        top_e_exp = top_e.pow(exp_frac)
        top_e_exp = torch.mul(top_e_exp, mask_top_e)

        mid_support = torch.mm(top_v,
                               torch.diag_embed(top_e_exp))
        fin_support = torch.mm(mid_support, top_v.t())
        if saveFile:
            with open(support_file_path, 'wb') as f_support:
                pickle.dump(fin_support, f_support, protocol=4)
    else:
        with open(support_file_path, 'rb') as f_support:
            fin_support = pickle.load(f_support)

    support = fin_support.to(device)

    # print("len adj_list:", len(support_list))
    return support, top_e, top_v


def _symeig_v2(adj):
    # The eigenvalues are returned in ascending order!
    e, v = torch.linalg.eigh(adj, "U")
    # e, v = torch.linalg.eigh(adj)
    e = _flip(e, 0)  # 利用镜像对称函数将所得升序特征值改为降序
    v = _flip(v, 1)  # 特征向量是按列存放的

    top_e = e
    top_v = v
    return top_e, top_v
