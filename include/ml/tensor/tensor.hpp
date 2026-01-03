#pragma once
#include <stdlib.h>
#include <vector>
#include "../autograd/base.hpp"

struct Storage {
	std::vector<float> data;
};



class Tensor {
	// --- memory ---
	std::shared_ptr<Storage> data;
	size_t offset;

	// --- view ---
	std::vector<int> sizes;
	std::vector<int> strides;

	// --- autograd ---
	bool requires_grad;
	std::unique_ptr<Tensor> grad;
	std::shared_ptr<GradFn> grad_fn;


};