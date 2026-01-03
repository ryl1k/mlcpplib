#include "ml/ops/matmul.hpp"
#include "ml/core/error.hpp"

namespace ml::ops {

    Tensor matmul(const Tensor& a, const Tensor& b) {
        ML_CHECK(a.ndim() == 2, "matmul: a must be 2D");
        ML_CHECK(b.ndim() == 2, "matmul: b must be 2D");
        ML_CHECK(a.sizes()[1] == b.sizes()[0], "matmul: shape mismatch");

        size_t M = a.sizes()[0];
        size_t K = a.sizes()[1];
        size_t N = b.sizes()[1];

        Tensor out = Tensor::zeros({ M, N });

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float aik = a.at({ i, k });
                for (size_t j = 0; j < N; ++j) {
                    out.at({ i, j }) += aik * b.at({ k, j });
                }
            }
        }
        return out;
    }

} // namespace ml::ops
