#include "ml/tensor/tensor.hpp"
#include "ml/core/error.hpp"

#include <algorithm> // fill, copy

namespace ml {

    // -------- internal ctor --------
    Tensor::Tensor(std::shared_ptr<Storage> storage,
        size_t offset,
        std::vector<size_t> sizes,
        std::vector<size_t> strides)
        : storage_(std::move(storage)),
        offset_(offset),
        sizes_(std::move(sizes)),
        strides_(std::move(strides)) {

        ML_CHECK(storage_ != nullptr, "Tensor: storage is null");
        ML_CHECK_EQ(sizes_.size(), strides_.size(), "Tensor: sizes/strides rank mismatch");

        // v1 restriction: no zero-sized dims
        for (size_t s : sizes_) {
            ML_CHECK(s > 0, "Tensor: dimension must be > 0 (v1 restriction)");
        }
    }

    // -------- factories --------
    Tensor Tensor::empty(const std::vector<size_t>& sizes) {
        auto st = std::make_shared<Storage>(core::numel(sizes));
        auto strides = core::contiguous_strides(sizes);
        return Tensor(st, 0, sizes, strides);
    }

    Tensor Tensor::zeros(const std::vector<size_t>& sizes) {
        Tensor t = empty(sizes);
        std::fill(t.storage_->data.begin(), t.storage_->data.end(), 0.0f);
        return t;
    }

    Tensor Tensor::ones(const std::vector<size_t>& sizes) {
        Tensor t = empty(sizes);
        std::fill(t.storage_->data.begin(), t.storage_->data.end(), 1.0f);
        return t;
    }

    Tensor Tensor::arange(size_t n) {
        Tensor t = empty({ n });
        for (size_t i = 0; i < n; ++i) {
            t.storage_->data[i] = static_cast<float>(i);
        }
        return t;
    }

    Tensor Tensor::from_vector(const std::vector<float>& v,
        const std::vector<size_t>& sizes) {
        ML_CHECK_EQ(v.size(), core::numel(sizes), "from_vector: data size != numel(shape)");
        auto st = std::make_shared<Storage>(v.size());
        std::copy(v.begin(), v.end(), st->data.begin());
        return Tensor(st, 0, sizes, core::contiguous_strides(sizes));
    }

    // -------- info --------
    size_t Tensor::ndim() const { return sizes_.size(); }
    size_t Tensor::numel() const { return core::numel(sizes_); }
    const std::vector<size_t>& Tensor::sizes() const { return sizes_; }
    const std::vector<size_t>& Tensor::strides() const { return strides_; }
    bool Tensor::is_contiguous() const { return core::is_contiguous(sizes_, strides_); }

    // -------- raw data --------
    float* Tensor::data() { return storage_->ptr() + offset_; }
    const float* Tensor::data() const { return storage_->ptr() + offset_; }

    const std::shared_ptr<Storage>& Tensor::storage_ptr() const { return storage_; }

    // -------- indexing (initializer_list) --------
    float& Tensor::at(std::initializer_list<size_t> idx_list) {
        std::vector<size_t> idx(idx_list.begin(), idx_list.end());
        return at_vec_(idx);
    }

    float Tensor::at(std::initializer_list<size_t> idx_list) const {
        std::vector<size_t> idx(idx_list.begin(), idx_list.end());
        return at_vec_(idx);
    }

    // -------- private helpers for indexing --------
    float& Tensor::at_vec_(const std::vector<size_t>& idx) {
        ML_CHECK_EQ(idx.size(), ndim(), "at(): wrong number of indices");
        for (size_t d = 0; d < idx.size(); ++d) {
            ML_CHECK_LT(idx[d], sizes_[d], "at(): index out of range");
        }

        size_t lin = core::linear_index(offset_, strides_, idx);
        ML_CHECK_LT(lin, storage_->size(), "at(): linear index out of storage bounds");
        return storage_->data[lin];
    }

    float Tensor::at_vec_(const std::vector<size_t>& idx) const {
        ML_CHECK_EQ(idx.size(), ndim(), "at() const: wrong number of indices");
        for (size_t d = 0; d < idx.size(); ++d) {
            ML_CHECK_LT(idx[d], sizes_[d], "at() const: index out of range");
        }

        size_t lin = core::linear_index(offset_, strides_, idx);
        ML_CHECK_LT(lin, storage_->size(), "at() const: linear index out of storage bounds");
        return storage_->data[lin];
    }

    // "odometer" increment: returns false when finished
    bool Tensor::next_index_(std::vector<size_t>& idx, const std::vector<size_t>& sizes) {
        // increment last dimension, carry if overflow
        for (size_t d = sizes.size(); d-- > 0; ) {
            idx[d] += 1;
            if (idx[d] < sizes[d]) return true; // no carry needed
            idx[d] = 0; // carry
        }
        return false; // overflowed past the first dim => done
    }

    // -------- views --------
    Tensor Tensor::reshape(const std::vector<size_t>& new_sizes) const {
        ML_CHECK(is_contiguous(), "reshape(): requires contiguous tensor (v1)");
        ML_CHECK_EQ(core::numel(new_sizes), numel(), "reshape(): numel mismatch");
        return Tensor(storage_, offset_, new_sizes, core::contiguous_strides(new_sizes));
    }

    Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
        ML_CHECK_LT(dim0, ndim(), "transpose(): dim0 out of range");
        ML_CHECK_LT(dim1, ndim(), "transpose(): dim1 out of range");

        auto new_sizes = sizes_;
        auto new_strides = strides_;
        std::swap(new_sizes[dim0], new_sizes[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        return Tensor(storage_, offset_, std::move(new_sizes), std::move(new_strides));
    }

    Tensor Tensor::slice(size_t dim, size_t start, size_t length) const {
        ML_CHECK_LT(dim, ndim(), "slice(): dim out of range");
        ML_CHECK(start + length <= sizes_[dim], "slice(): range out of bounds");

        auto new_sizes = sizes_;
        new_sizes[dim] = length;

        size_t new_offset = offset_ + start * strides_[dim];
        return Tensor(storage_, new_offset, std::move(new_sizes), strides_);
    }

    // -------- contiguous materialize --------
    Tensor Tensor::contiguous() const {
        if (is_contiguous()) {
            // return an equivalent view (no copy)
            return Tensor(storage_, offset_, sizes_, strides_);
        }

        Tensor out = empty(sizes_);

        if (ndim() == 0) {
            // scalar
            out.storage_->data[0] = storage_->data[offset_];
            return out;
        }

        std::vector<size_t> idx(ndim(), 0);

        for (size_t out_lin = 0; out_lin < out.numel(); ++out_lin) {
            out.storage_->data[out_lin] = at_vec_(idx);
            if (!next_index_(idx, sizes_)) break;
        }

        return out;
    }

} // namespace ml
