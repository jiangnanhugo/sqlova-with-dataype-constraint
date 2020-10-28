# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
import os
from matplotlib.pylab import *


def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv


def ensure_dir(my_path):
    """ Generate directory if not exists
    """
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def topk_multi_dim(tensor, n_topk=1, batch_exist=True):
    if batch_exist:
        # tensor: [bS, beam_size, n_agg_ops]
        idxs = []
        for b, tensor1 in enumerate(tensor):      # [beam_size, n_agg_ops]
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)     # [beam_size * n_agg_ops]
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            # get the index (index of sel_col in beam, index of agg_op) of top_k elements (separately)
            # (dim0, dim1, dim2, ...)

            # reconstruct: [[sc_idx_1, ag_idx_1], ..., [sc_idx_k, ag_idx_k]]
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)

    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = unravel_index(idxs_1d.numpy(), tensor1.shape)
        # (dim0, dim1, dim2, ...)

        # reconstruct
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs


def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int64): return int(o)
    raise TypeError
