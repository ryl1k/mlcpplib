#pragma once
#include "ml/tensor/tensor.hpp"

namespace ml::ops {

	Tensor add(const Tensor& a, const Tensor& b);
	Tensor sub(const Tensor& a, const Tensor& b);
	Tensor mul(const Tensor& a, const Tensor& b);

	Tensor relu(const Tensor& x);

}