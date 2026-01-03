#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ml/core/shape.hpp"
#include "ml/tensor/tensor.hpp"

static void expect_throw(const char* name, const std::function<void()>& fn) {
    try {
        fn();
        std::cerr << "[FAIL] Expected exception: " << name << "\n";
        std::abort();
    }
    catch (const std::exception&) {
        std::cout << "[OK]   threw: " << name << "\n";
    }
}

int main() {
    using namespace ml;

    std::cout << "Running tests...\n";

    // ---- shape tests ----
    {
        assert(ml::core::numel({ 2,3,4 }) == 24);
        auto s = ml::core::contiguous_strides({ 2,3,4 });
        assert((s == std::vector<size_t>{12, 4, 1}));
        assert(ml::core::is_contiguous({ 2,3,4 }, { 12,4,1 }));
        assert(!ml::core::is_contiguous({ 2,3,4 }, { 1,4,12 }));
        assert(ml::core::linear_index(0, { 12,4,1 }, { 1,2,3 }) == 23);
        std::cout << "[OK]   shape basics\n";
    }

    // ---- tensor creation + indexing ----
    {
        auto v = Tensor::arange(6);               // [0 1 2 3 4 5]
        auto A = v.reshape({ 2,3 });                // [[0 1 2],[3 4 5]]
        assert(A.ndim() == 2);
        assert(A.numel() == 6);
        assert(A.is_contiguous());

        assert(A.at({ 0,0 }) == 0.0f);
        assert(A.at({ 0,2 }) == 2.0f);
        assert(A.at({ 1,0 }) == 3.0f);
        assert(A.at({ 1,2 }) == 5.0f);

        A.at({ 1,1 }) = 42.0f;
        assert(A.at({ 1,1 }) == 42.0f);

        std::cout << "[OK]   tensor create + at()\n";
    }

    // ---- transpose is a view (no copy) ----
    {
        auto A = Tensor::arange(6).reshape({ 2,3 }); // [[0 1 2],[3 4 5]]
        auto B = A.transpose(0, 1);                 // shape [3,2]

        // same storage => no copy
        assert(A.storage_ptr() == B.storage_ptr());

        // index mapping check: A(i,j) == B(j,i)
        assert(A.at({ 1,2 }) == B.at({ 2,1 }));
        assert(A.at({ 0,1 }) == B.at({ 1,0 }));

        // transpose usually makes tensor non-contiguous
        assert(!B.is_contiguous());

        std::cout << "[OK]   transpose view\n";
    }

    // ---- slice is a view (offset changes) ----
    {
        auto A = Tensor::arange(6).reshape({ 2,3 }); // [[0 1 2],[3 4 5]]
        auto row1 = A.slice(0, 1, 1);              // second row, shape [1,3]

        assert(A.storage_ptr() == row1.storage_ptr()); // no copy
        assert((row1.sizes() == std::vector<size_t>{1, 3}));
        assert(row1.at({ 0,0 }) == A.at({ 1,0 }));
        assert(row1.at({ 0,2 }) == A.at({ 1,2 }));

        std::cout << "[OK]   slice view\n";
    }

    // ---- contiguous materialize ----
    {
        auto A = Tensor::arange(6).reshape({ 2,3 });
        auto Bt = A.transpose(0, 1);
        auto Bc = Bt.contiguous();

        assert(Bc.is_contiguous());
        assert(Bc.storage_ptr() != Bt.storage_ptr()); // copy happened
        assert((Bc.sizes() == Bt.sizes()));            // same logical shape

        // values preserved
        for (size_t i = 0; i < Bc.sizes()[0]; ++i) {
            for (size_t j = 0; j < Bc.sizes()[1]; ++j) {
                assert(Bc.at({ i,j }) == Bt.at({ i,j }));
            }
        }

        std::cout << "[OK]   contiguous copy\n";
    }

    // ---- error cases ----
    {
        auto A = Tensor::arange(6).reshape({ 2,3 });

        expect_throw("at() out of range", [&] { (void)A.at({ 100,0 }); });
        expect_throw("reshape numel mismatch", [&] { (void)A.reshape({ 5,5 }); });
        expect_throw("transpose bad dim", [&] { (void)A.transpose(0, 10); });
        expect_throw("slice out of bounds", [&] { (void)A.slice(0, 2, 2); });

        std::cout << "[OK]   error cases\n";
    }

    std::cout << "All tests passed ✅\n";
    return 0;
}
