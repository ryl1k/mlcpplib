#pragma once
#include <memory>
#include <vector>

namespace ml {

    class Tensor;

    // graph knows how to do backward for some op
    struct GradFn {
        virtual ~GradFn() = default;

        // grad_out = dL/d(this_output)
        virtual void backward(const Tensor& grad_out) = 0;
    };

} // namespace ml
