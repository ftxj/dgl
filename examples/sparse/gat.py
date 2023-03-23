"""
[Graph Attention Networks]
(https://arxiv.org/abs/1710.10903)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, FlickrDataset
from torch.optim import Adam
import ctypes
import python_ops

class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def foo2(self, A):
        return A.row, A.col
    
    def foo3(self, A):
        return A.val

    def forward(self, A_hat, Z):
        
        row, col = self.foo2(A_hat)

        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[row] + e_r[col]

        a = F.leaky_relu(e)
        tmp = dglsp.val_like(A_hat, a)
        # A_atten = tmp.softmax()
        A_atten = python_ops.py_softmax(tmp)

        # print(torch.allclose(A_atten.indices(), tmp2.indices()))
        # print(torch.allclose(A_atten.val, tmp2.val))

        # exit()

        a_drop = self.dropout(self.foo3(A_atten))
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)


class GAT(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6
    ):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A, X):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(A, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(A, Z).mean(-1)
        return Z


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc



def train(model, g, A, X):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(50):
        # Forward.
        model.train()


        if(epoch == 10):
            _cudart = ctypes.CDLL('libcudart.so')
            ret = _cudart.cudaProfilerStart()

            
        torch.cuda.nvtx.range_push("forward" + str(epoch))
        logits = model(A, X)
        torch.cuda.nvtx.range_pop()

        if(epoch == 49):
            ret = _cudart.cudaProfilerStop()

        # # Compute loss with nodes in training set.
        # loss = F.cross_entropy(logits[train_mask], label[train_mask])

        # # Backward.
        # optimizer.zero_grad()
        # # loss.backward()
        # optimizer.step()

        # Compute prediction.
        # model.eval()
        # logits = model(A_hat, X)
        # pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        # val_acc, test_acc = evaluate(g, pred)
        # print(
        #     f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
        #     f" acc: {test_acc:.3f}"
        # )
        print(epoch)


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = FlickrDataset()
    g = dataset[0].to(dev)

    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Add self-loops.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I

    # Create GAT model.
    X = g.ndata["feat"]
    in_size = X.shape[1]
    out_size = dataset.num_classes
    model = GAT(in_size, out_size).to(dev)
    from torch._dynamo import config
    config.suppress_errors = True

    import logging
    from torch._inductor import config
    config.debug = True

    # from torch._dynamo import config as config2
    # config2.log_level = logging.DEBUG
    # config2.output_code = True

    model = torch.compile(model)
    # Kick off training.
    train(model, g, A_hat, X)
