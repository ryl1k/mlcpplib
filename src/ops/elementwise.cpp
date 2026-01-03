#include "ml/ops/elementwise.hpp"
#include "ml/core/error.hpp"

namespace ml::ops{

	static void check_same_shape(const Tensor& a, const Tensor& b) {
		ML_CHECK(a.sizes() == b.sizes(), "elementwise: shape mismatch");
	}

	Tensor add(const Tensor& a, const Tensor& b) {
		check_same_shape(a, b);

		Tensor out = Tensor::empty(a.sizes());

		for (size_t i = 0; i < out.numel(); ++i) {
			out.data()[i] = a.data()[i] + b.data()[i];
		}
		return out;
	}

	Tensor sub(const Tensor& a, const Tensor& b) {
		check_same_shape(a, b);
		Tensor out = Tensor::empty(a.sizes());

		for (size_t i = 0; i < out.numel(); ++i) {
			out.data()[i] = a.data()[i] - b.data()[i];
		}
		return out;
	}

	Tensor mul(const Tensor& a, const Tensor& b) {
		check_same_shape(a, b);
		Tensor out = Tensor::empty(a.sizes());

		for (size_t i = 0; i < out.numel(); ++i) {
			out.data()[i] = a.data()[i] * b.data()[i];
		}
		return out;
	}

	Tensor relu(const Tensor& x) {
		Tensor out = Tensor::empty(x.sizes());

		for (size_t i = 0; i < out.numel(); ++i) {
			float v = x.data()[i];
			out.data()[i] = (v > 0.0f) ? v : 0.0f;
		}
		return out;
	}

}