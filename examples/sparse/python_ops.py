import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


def py_BroadcastOpNoAutoGrad(sparse_mat: dglsp.SparseMatrix, dense_mat: torch.Tensor, op, dim: int):
    sparse_val = sparse_mat.val
    out_row = sparse_mat.nnz
    shape = [out_row, sparse_val.size(1)]
    ret = torch.zeros(shape, dtype = sparse_val.dtype, device = sparse_val.device)
    return torch.ops.dgl_sparse.broadcast_op_unfused_part0(sparse_mat.c_sparse_matrix, sparse_val, dense_mat, ret, op, dim)

def py_BroadcastSubNoAutoGrad(sparse_mat, dense_mat, dim):
    return py_BroadcastOpNoAutoGrad(sparse_mat, dense_mat, "sub", dim);

def py_BroadcastDivNoAutoGrad(sparse_mat, dense_mat, dim):
    return py_BroadcastOpNoAutoGrad(sparse_mat, dense_mat, "div", dim);


def py_reduce_along(A, reduce, dim):
    value = A.val
    coo = A.coo()

    indices = A.indices()

    if (reduce == "sum"):
        reduce_op = "sum"
    elif (reduce == "smin"):
        reduce_op = "amin"
    elif (reduce == "smax"):
        reduce_op = "amax"

    output_shape = list(value.size())
    view_dims = [1] * len(output_shape)
    view_dims[0] = -1
    if (dim == 0):
        output_shape[0] = len(coo[0])
        idx = indices[1].view(view_dims).expand_as(value)
    elif (dim == 1):
        output_shape[0] = len(coo[1])
        idx = indices[0].view(view_dims).expand_as(value)
    out = torch.zeros(output_shape, dtype = value.dtype, device = value.device)
    if (dim == 0):
        out.scatter_reduce_(0, idx, value, reduce_op, include_self=False)
    elif (dim == 1):
        out.scatter_reduce_(0, idx, value, reduce_op, include_self=False)
    return out

def py_reduce_min(A, dim):
    return py_reduce_along(A, "smin", dim)

def py_reduce_max(A, dim):
    return py_reduce_along(A, "smax", dim)

def py_reduce_sum(A, dim):
    return py_reduce_along(A, "sum", dim)

def py_softmax_forward(sparse_mat, sparse_val, dim):
    sparse_val_max = py_reduce_max(sparse_mat, dim)
    sparse_val_exp = py_BroadcastSubNoAutoGrad(sparse_mat, sparse_val_max, dim).exp()
    sparse_val_sum = py_reduce_sum(dglsp.val_like(sparse_mat, sparse_val_exp), dim)
    sparse_score = py_BroadcastDivNoAutoGrad(dglsp.val_like(sparse_mat, sparse_val_exp), sparse_val_sum, dim)
    sparse_requires_grad = sparse_val.requires_grad
    if sparse_requires_grad:
        cache_sparse_score = sparse_score

    # ctx->saved_data["sparse_matrix"] = sparse_mat;
    # ctx->saved_data["sparse_requires_grad"] = sparse_requires_grad;
    # ctx->saved_data["dim"] = dim;
    # ctx->save_for_backward({cache_sparse_score});
    return sparse_score

def py_softmax(sparse_mat: dglsp.SparseMatrix , dim: int = 1):
    sparse_val = sparse_mat.val
    expand_dim = False
    new_sparse_mat = sparse_mat
    if (sparse_val.dim() == 1):
        sparse_val = sparse_val.view([-1, 1])
        expand_dim = True
        new_sparse_mat = dglsp.val_like(sparse_mat, sparse_val)

    new_sparse_val = py_softmax_forward(new_sparse_mat, sparse_val, dim)

    if expand_dim:
        new_sparse_val = new_sparse_val.view(-1)
    return dglsp.val_like(sparse_mat, new_sparse_val)


indices = torch.tensor([[0, 1, 1], [0, 0, 2]])
val = torch.tensor([1, 1, 2])

A = dglsp.spmatrix(indices, val, shape=(4, 3))

res = dglsp.smin(A, 0)
res2 = py_reduce_min(A, 0)
