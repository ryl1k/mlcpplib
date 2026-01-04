#pragma once
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "ml/autograd/grad_fn.hpp"
#include "ml/core/storage.hpp"
#include "ml/core/shape.hpp"

namespace ml {

    class Tensor {
    public:
        // --- factories ---
        static Tensor empty(const std::vector<size_t>& sizes);
        static Tensor zeros(const std::vector<size_t>& sizes);
        static Tensor ones(const std::vector<size_t>& sizes);
        static Tensor arange(size_t n);
        static Tensor from_vector(const std::vector<float>& v,
            const std::vector<size_t>& sizes);

        // --- info ---
        size_t ndim() const;
        size_t numel() const;
        const std::vector<size_t>& sizes() const;
        const std::vector<size_t>& strides() const;
        bool is_contiguous() const;

        // --- raw data ---
        float* data();
        const float* data() const;

        // --- indexing ---
        float& at(std::initializer_list<size_t> idx);
        float  at(std::initializer_list<size_t> idx) const;

        // --- autograd flags ---
        bool requires_grad() const { return requires_grad_; }
        void set_requires_grad(bool v) { requires_grad_ = v; }

        // grad can not exist
        bool has_grad() const { return (bool)grad_; }
        const Tensor& grad() const;     // throw if none
        Tensor& grad_mut();             // throw if none
        void zero_grad();               // if grad exists - fill with zeros

        // start backward as loss
        // (scalar loss)
        void backward();                // easy v1

        // internal: set graph node
        void set_grad_fn(std::shared_ptr<GradFn> fn) { grad_fn_ = std::move(fn); }
        std::shared_ptr<GradFn> grad_fn() const { return grad_fn_; }

        // factories that help autograd
        static Tensor zeros_like(const Tensor& t);
        static Tensor ones_like(const Tensor& t);

        // --- views ---
        Tensor reshape(const std::vector<size_t>& new_sizes) const;
        Tensor transpose(size_t dim0, size_t dim1) const;
        Tensor slice(size_t dim, size_t start, size_t length) const;

        // --- materialize ---
        Tensor contiguous() const;

        // for tests/debug: do two tensors share the same buffer?
        const std::shared_ptr<Storage>& storage_ptr() const;

    private:
        // internal constructor (used for views)
        Tensor(std::shared_ptr<Storage> storage,
            size_t offset,
            std::vector<size_t> sizes,
            std::vector<size_t> strides);

        float  at_vec_(const std::vector<size_t>& idx) const;
        float& at_vec_(const std::vector<size_t>& idx);
        static bool next_index_(std::vector<size_t>& idx, const std::vector<size_t>& sizes);


        std::shared_ptr<Storage> storage_;
        size_t offset_{ 0 };
        std::vector<size_t> sizes_;
        std::vector<size_t> strides_;

        // --- autograd metadata ---
        bool requires_grad_{ false };
        std::unique_ptr<Tensor> grad_;           // alocate if needed
        std::shared_ptr<GradFn> grad_fn_;        // can be null 
    };

} // namespace ml
