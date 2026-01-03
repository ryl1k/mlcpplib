#pragma once
#include <cstddef>
#include <vector>

namespace ml::core {

    // number of elements
    // [] -> 1 (scalar)
    size_t numel(const std::vector<size_t>& sizes);

    // compute row-major contiguous strides
    // sizes [2,3,4] -> strides [12,4,1]
    std::vector<size_t> contiguous_strides(const std::vector<size_t>& sizes);

    // true if strides match contiguous_strides(sizes)
    bool is_contiguous(const std::vector<size_t>& sizes,
        const std::vector<size_t>& strides);

    // offset + sum(indices[d] * strides[d])
    size_t linear_index(size_t offset,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& indices);

} // namespace ml::core
