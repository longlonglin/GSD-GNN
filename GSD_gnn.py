import dgl.function as fn
import numpy as np
import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utilsforme import MixedDropout,torch_to_matrix,matrix_to_torch,filter_sparse_tensor
import torch.nn.functional as F
from scipy.special import iv
import time
import random
def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:
        masks = th.bernoulli(1.0 - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1.0 - drop_rate)

    return feats


class MLP(nn.Module):
    def __init__(
        self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False
    ):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.input_dropout = nn.Dropout(input_droprate)
        self.hidden_dropout = nn.Dropout(hidden_droprate)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


class GSD_gnn(nn.Module):
    r"""

    Parameters
    -----------
    in_dim: int
        Input feature size. 
    hid_dim: int
        Hidden feature size.
    n_class: int
        Number of classes.
    S: int
        Number of perturbations
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLPx
    batchnorm: bool, optional
        If True, use batch normalization.

    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        n_class,
        S=1,
        node_dropout=0.0,
        input_droprate=0.0,
        hidden_droprate=0.0,
        batchnorm=False,
            ):
        super(GSD_gnn, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.n_class = n_class

        self.mlp = MLP(
            in_dim, hid_dim, n_class, input_droprate, hidden_droprate, batchnorm
        )

        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

        self.edge_dropout = MixedDropout(0.3)   
   
        
    def forward(self, feats,p, training=True):
        X = feats
        S = self.S
        

        if training:  # Training Mode
            output_list = []
            
            for s in range(S):
                if random.random() < 0.5:
                    drop_feat = drop_node(X, self.dropout, True) # Drop nodes
                    feat =  p@drop_feat   # 
                    output_list.append(
                        th.log_softmax(self.mlp(feat), dim=-1)
                    )  # Prediction
                else:
                    feat =  self.edge_dropout(p)@X   # drop edges
                    output_list.append(
                    th.log_softmax(self.mlp(feat), dim=-1)
                )  # Prediction
            return output_list
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            feat = p@drop_feat 
            return th.log_softmax((self.mlp(feat)), dim=-1)
        
    