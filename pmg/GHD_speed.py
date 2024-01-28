import numpy as np
import networkx as nx
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

from tqdm import tqdm
import time
import random
from .Utils import alias_draw, alias_setup
from .baseline.BaseModel import BaseModel
import multiprocessing as mp
import scipy.sparse as sp
import dgl
import math
from .Utils import matrix_to_torch,get_topk_rows,get_topklist
import gc
from scipy.sparse import csr_matrix, save_npz

from numpy import int32 
from numpy import float32

from sampler import walker


class ghd(BaseModel):
   
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--window-size", type=int32, default=10,
                            help="Window size of approximate matrix. Default is 10.")
        
        parser.add_argument("--num-round", type=int32, default=100,
                            help="Number of round c. Default is 100.")
        parser.add_argument("--hidden-size", type=int32, default=128)
        parser.add_argument("--a_decay", type=int32, default=0.1)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.window_size, args.num_round, args.a_decay)

    def __init__(self, w, window_size, num_round,p,r):
        super(ghd, self).__init__()
        self.w = w
        self.r=r
        self.window_size = window_size
        self.p = p
        self.num_round = num_round

  
    def forward(self, adj,if_undirc=True,flag=False):
       
        self.G = adj.to('cpu')
        self.adj = self.G.adjacency_matrix()
        
        A =  sp.csr_matrix((self.adj.val.numpy(),(self.adj.indices()[0].numpy(),self.adj.indices()[1].numpy())),shape=self.adj.shape)
        degrees_list = np.array(A.sum(axis=0))[0]
 
        indices= walker.UIntVector((A.indices.astype(np.uint32)))
        indptr = walker.UIntVector((A.indptr.astype(np.uint32)))
        
        self.num_node = self.G.num_nodes() 
        self.num_edge = self.G.num_edges()  
        self.avg_degree = int(self.num_edge/self.num_node)
        print("edge_num:",self.num_edge)
        print("node_num:",self.num_node)
        
        #********************Run in C++here to obtain the sampled matrix**********************************
        stime = time.time()
        csr_matrix= walker.BinaryGraphWalker.run(indices_=indices,indptr_=indptr,num_node_=self.num_node, num_round=self.num_round,window_size=self.window_size,w=self.w,p=self.p)
        etime = time.time()
        sample_time = etime-stime
        print('sample time :',sample_time)
        
        
        stime = time.time()
        matrix = sp.csr_matrix((self.num_node , self.num_node))
        matrix.data=np.fromiter(csr_matrix.data, dtype=float)                                         #np.array(csr_matrix.data)
        matrix.indices=np.fromiter(csr_matrix.indices, dtype=int)                                      #np.array(csr_matrix.indices)
        matrix.indptr = np.fromiter(csr_matrix.indptr, dtype=int)                                     #np.array(csr_matrix.indptr)
        etime = time.time()
        print('csr_matrix time :',etime-stime)
        stime = time.time()
        I_identity = sp.identity(self.num_node,dtype=float, format='csr')  
        print('chech point 1')
        degree = sp.diags(np.array(A.sum(axis=0))[0], format="csr",dtype=float)  # 
     
        print('chech point 2')
        degree_inv1 = degree.power(self.r-1)
        print('chech point 3')
        degree_inv2 = degree.power(-self.r)
        print('chech point 4')
        
        
        
        L_ = sp.csgraph.laplacian(matrix, normed=False, return_diag=False)
        print('chech point 5')
        ar = []  # 
        for i in range(0, self.window_size + 1):  # 
            ar.append(pow(self.w, i)/(pow(math.factorial(i), self.p)))  # 
        ar = np.array(ar)

        ar = ar / sum(ar)  # 
        ar = list(ar)
        theat0 = ar.pop(0)
        sum_ =  sum(ar)  # 
        print('chech point 6')
        gdm = (sum_ * (I_identity- degree_inv1.dot(L_).dot(degree_inv2)) + theat0*I_identity) 
        print('chech point 7')
        # 
        print(' Before Node-wise  :',len(gdm .data))
        topklist = get_topklist(degrees_list,min_k=self.avg_degree)
        M = get_topk_rows(gdm, topklist)
        print('After Node-wise  :',len(M.data))
        etime = time.time()
        construct_time = etime-stime
        print('construct time :',construct_time)
        print('total time:',construct_time+sample_time)
        preprocessing.normalize(M, "l1") 
        M = matrix_to_torch(M)
        
       
        return M
        
    
 