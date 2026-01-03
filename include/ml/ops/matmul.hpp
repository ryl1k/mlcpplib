#pragma once
#include "ml/tensor/tensor.hpp"

namespace ml::ops {

	Tensor matmul(const Tensor& a, const Tensor& b);

}
