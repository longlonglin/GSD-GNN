import argparse
import warnings
from pmg.GHD_speed import ghd
import dgl
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from GSD_gnn import GSD_gnn

from pmg.GHD_speed import ghd

warnings.filterwarnings("ignore")


def argument():
    parser = argparse.ArgumentParser()

    # data source params
    parser.add_argument(
        "--dataname", type=str, default="cora", help="Name of dataset."
    )
    # cuda params
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index. Default: -1, using CPU."
    )
    # training params
    parser.add_argument(
        "--epochs", type=int, default=200, help="Training epochs."
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=200,
        help="Patient epochs to wait before early stopping.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="L2 reg."
    )
    # model params
    parser.add_argument(
        "--hid_dim", type=int, default=32, help="Hidden layer dimensionalities."
    )
    parser.add_argument(
        "--dropnode_rate",
        type=float,
        default=0.5,
        help="Dropnode rate (1 - keep probability).",
    )
    parser.add_argument(
        "--input_droprate",
        type=float,
        default=0.0,
        help="dropout rate of input layer",
    )
    parser.add_argument(
        "--hidden_droprate",
        type=float,
        default=0.0,
        help="dropout rate of hidden layer",
    )
    parser.add_argument(
        "--sample", type=int, default=4, help="Sampling times of drop"
    )

   
    parser.add_argument(
        "--use_bn",
        action="store_true",
        default=False,
        help="Using Batch Normalization",
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    return args


if __name__ == "__main__":
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # Load from DGL dataset
    args = argument()
    print(args)

    if args.dataname == "cora":
        dataset = CoraGraphDataset()
    elif args.dataname == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataname == "pubmed":
        dataset = PubmedGraphDataset()
    graph = dataset[0]

    graph = dgl.add_self_loop(graph)
    device = args.device

    # retrieve the number of classes
    n_classes = dataset.num_classes

    # retrieve labels of ground truth
    labels = graph.ndata.pop("label").to(device).long()

    # Extract node features
    feats = graph.ndata.pop("feat").to(device)
    n_features = feats.shape[-1]
    
    # get progation matrix
    P_matrix_getter = ghd(w=5,window_size=20,num_round=96*50,worker=60,p=1.5,r=0.6)
    P_matrix= (P_matrix_getter(graph)).to('cuda:0')
    
    

    # retrieve masks for train/validation/test
    train_mask = graph.ndata.pop("train_mask")
    val_mask = graph.ndata.pop("val_mask")
    test_mask = graph.ndata.pop("test_mask")

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze().to(device)
    graph = graph.to(args.device)
    # Step 2: Create model =================================================================== #
    model = GSD_gnn(
        n_features,
        args.hid_dim,
        n_classes,
        args.sample,
        args.dropnode_rate,
        args.input_droprate,
        args.hidden_droprate,
        args.use_bn,
    )

    model = model.to(args.device)
    

    # Step 3: Create training components ===================================================== #
    loss_fn = nn.NLLLoss()
    opt = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_best = np.inf
    acc_best = 0
    loss_fcn = nn.CrossEntropyLoss()
    # Step 4: training epoches =============================================================== #
    for epoch in range(args.epochs):
        """Training"""
        model.train()
        t0 = time.time()
        loss_sup = 0
        logits = model(feats,P_matrix, True)         
        for k in range(args.sample):  # 
            
            loss_sup += loss_fcn(logits[k][train_idx], labels[train_idx])
        
            
            
        loss_sup = loss_sup / args.sample


        loss_train = loss_sup
        acc_train = th.sum(
            logits[0][train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)

        # backward
        opt.zero_grad()
        loss_train.backward()
        opt.step()
        mean=0
        """ Validating """
        model.eval()
        with th.no_grad():
            val_logits = model(feats,P_matrix, False) 
            loss_val = loss_fcn(val_logits[val_idx], labels[val_idx])
            acc_val = th.sum(
                val_logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)

            # Print out performance
            mean =  (time.time() - t0)
            # Print out performance
            print(
                "In epoch {}, epoch time:{:.4f},Train Acc: {:.4f} | Train Loss: {:.4f} ,Val Acc: {:.4f} | Val Loss: {:.4f}".format(
                    epoch,
                    mean,
                    acc_train,
                    loss_train.item(),
                    acc_val,
                    loss_val.item(),
                    )
                )

            # set early stopping counter
            if loss_val < loss_best or acc_val > acc_best:
                if loss_val < loss_best:
                    best_epoch = epoch
                    th.save(model.state_dict(), args.dataname + ".pkl")
                no_improvement = 0
                loss_best = min(loss_val, loss_best)
                acc_best = max(acc_val, acc_best)
            else:
                no_improvement += 1
                if no_improvement == args.early_stopping:
                    print("Early stopping.")
                    break

    print("Optimization Finished!")

    print("Loading {}th epoch".format(best_epoch))
    model.load_state_dict(th.load(args.dataname + ".pkl"))

    """ Testing """
    model.eval()

    test_logits = model(feats,P_matrix, False) #
    test_acc = th.sum(
        test_logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)

    print("Test Acc: {:.4f}".format(test_acc))