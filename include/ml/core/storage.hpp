#pragma once
#include <vector>
#include <cstddef>

namespace ml::core {

    struct Storage {
        std::vector<float> data;

        explicit Storage(size_t n)
            : data(n) {
        }

        size_t size() const {
            return data.size();
        }

        float* ptr() {
            return data.data();
        }

        const float* ptr() const {
            return data.data();
        }
    };

}