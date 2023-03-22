#include <sparse/sparse_matrix.h>
#include <torch/script.h>

#include <string>

#include <sparse/dgl_headers.h>

#include <sparse/sparse_matrix.h>
#include <sparse/unfused_ops.h>
#include <torch/script.h>

#include "./utils.h"

namespace dgl {
namespace sparse {


torch::Tensor BroadcastOpNoAutoGrad_Unfused0(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor &sparse_val, 
    torch::Tensor &dense_mat, 
    torch::Tensor &ret,
    const std::string& op, int64_t dim) {

  auto dgl_sparse_val = TorchTensorToDGLArray(sparse_val);
  auto dgl_dense_mat = TorchTensorToDGLArray(dense_mat);
  auto dgl_ret = TorchTensorToDGLArray(ret);
  auto dgl_rhs_target = dim == 0 ? 2 : 0;

  if (sparse_mat->HasCOO() || !sparse_mat->HasCSR()) {
    auto coo = COOToOldDGLCOO(sparse_mat->COOPtr());
    aten::COOSDDMM(
        op.c_str(), coo, dgl_sparse_val, dgl_dense_mat, dgl_ret,
        1 /* Lhs target: e */, dgl_rhs_target);
  } else {
    auto csr = CSRToOldDGLCSR(sparse_mat->CSRPtr());
    aten::CSRSDDMM(
        op.c_str(), csr, dgl_sparse_val, dgl_dense_mat, dgl_ret,
        1 /* Lhs target: e */, dgl_rhs_target);
  }
  return ret;
}

}  // namespace sparse
}  // namespace dgl