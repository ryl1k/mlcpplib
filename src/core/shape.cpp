#include "ml/core/shape.hpp"
#include "ml/core/error.hpp"

namespace ml::core {

    size_t numel(const std::vector<size_t>& sizes) {
        size_t els = 1;
        for (size_t s : sizes) {
            ML_CHECK(s > 0, "numel(): dimension must be > 0 (v1 restriction)");
            els *= s;
        }
        return els;
    }

    std::vector<size_t> contiguous_strides(const std::vector<size_t>& sizes) {
        std::vector<size_t> strides(sizes.size(), 0);
        if (sizes.empty()) {
            return strides; // scalar: no dims
        }

        strides.back() = 1;
        for (size_t i = sizes.size() - 1; i-- > 0; ) {
            ML_CHECK(sizes[i + 1] > 0, "contiguous_strides(): dimension must be > 0");
            strides[i] = strides[i + 1] * sizes[i + 1];
        }
        return strides;
    }

    bool is_contiguous(const std::vector<size_t>& sizes,
        const std::vector<size_t>& strides) {
        if (sizes.size() != strides.size()) return false;
        return contiguous_strides(sizes) == strides;
    }

    size_t linear_index(size_t offset,
        const std::vector<size_t>& strides,
        const std::vector<size_t>& indices) {
        ML_CHECK_EQ(strides.size(), indices.size(), "linear_index(): rank mismatch");
        size_t idx = offset;
        for (size_t d = 0; d < indices.size(); ++d) {
            idx += indices[d] * strides[d];
        }
        return idx;
    }

} // namespace ml::core
