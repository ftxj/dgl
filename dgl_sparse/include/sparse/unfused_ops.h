#ifndef SPARSE_UNFUSED_OPS_H_
#define SPARSE_UNFUSED_OPS_H_

#include <sparse/sparse_matrix.h>

namespace dgl {
namespace sparse {

torch::Tensor BroadcastOpNoAutoGrad_Unfused0(
    const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor &sparse_val, torch::Tensor &dense_mat, torch::Tensor &ret,
    const std::string& op, int64_t dim);

}  // namespace sparse
}  // namespace dgl

#endif  // SPARSE_UNFUSED_OPS_H_
