#pragma once
#include <stdlib.h>
#include <vector>

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


	bool requires_grad;
	std::unique_ptr<Tensor> grad;



};