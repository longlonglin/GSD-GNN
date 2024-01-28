import errno
import itertools
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from urllib import request
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse

# from tabulate import tabulate
#
# from .graph_utils import coo2csr_index
from numba import jit

def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
    
def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)
    

# @jit(nopython=True)
def get_topk_rows(matrix, topk_list):
    rows, cols = matrix.shape
    new_data, new_row, new_col = [], [], []
    for row in range(rows):
        data = matrix.data[matrix.indptr[row]:matrix.indptr[row+1]]
        indices = matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
        topk = topk_list[row]
        
        if len(data) > topk:
            idx = np.argpartition(data, -topk)[-topk:]
            data = data[idx]
            indices = indices[idx]
     
        new_data.extend(data)
        new_row.extend([row]*len(data))
        new_col.extend(indices)
    return coo_matrix((new_data, (new_row, new_col)), shape=matrix.shape)


# @jit(nopython=True)
# def get_topk_rows_turbo(data,indices,topk):
           
#         idx = np.argpartition(data, -topk)[-topk:]
#         data = data[idx]
#         indices = indices[idx]

#         return  data,indices

@jit(nopython=True)
def get_topklist(degrees_list,min_k):
    
    topk_list = []
    for  v  in degrees_list:
        k = math.ceil(512*np.log2(2/(v+1)+1))
        if k >=min_k:
            topk_list.append(k)
        else:
            topk_list.append(min_k)

    return topk_list

