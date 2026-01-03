#pragma once

struct GradFn {
	virtual ~GradFn() = default;
	virtual void backward(const Tensor& grad_output) = 0;
};
