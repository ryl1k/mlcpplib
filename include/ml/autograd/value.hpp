#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stdexcept>

namespace ml::autograd {

    // Forward declaration
    struct Value;
    using V = std::shared_ptr<Value>;

    // one node for scalar
    struct Value {
        double data = 0.0;   // forward value
        double grad = 0.0;   // d(loss)/d(this)

        // parents in node
        std::vector<V> parents;

        // local backwards: how to give backward to parents
        // catch pointers for backward
        std::function<void()> backward_fn;

        explicit Value(double v) : data(v) {}

        // ====== API creation ======
        static V make(double v) { return std::make_shared<Value>(v); }

        // ====== backward start ======
        void backward() {
            // 1) topologic sort
            std::vector<Value*> topo;
            std::unordered_set<Value*> visited;
            topo_sort(this, topo, visited);

            // 2) start dl/dl = 1
            grad = 1.0;

            // 3) go backwards
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                Value* node = *it;
                if (node->backward_fn) node->backward_fn();
            }
        }

    private:
        static void topo_sort(Value* v,
            std::vector<Value*>& topo,
            std::unordered_set<Value*>& visited) {
            if (visited.count(v)) return;
            visited.insert(v);
            for (auto& p : v->parents) {
                topo_sort(p.get(), topo, visited);
            }
            topo.push_back(v);
        }
    };

    // ====== Ops (forward and backward) ======

    inline V add(const V& a, const V& b) {
        auto out = Value::make(a->data + b->data);
        out->parents = { a, b };

        // local df:
        // out = a + b
        // dL/da += dL/dout * 1
        // dL/db += dL/dout * 1
        out->backward_fn = [out, a, b]() {
            a->grad += 1.0 * out->grad;
            b->grad += 1.0 * out->grad;
            };
        return out;
    }

    inline V sub(const V& a, const V& b) {
        auto out = Value::make(a->data - b->data);
        out->parents = { a, b };
        // out = a - b
        // dL/da += dL/dout
        // dL/db += -dL/dout
        out->backward_fn = [out, a, b]() {
            a->grad += 1.0 * out->grad;
            b->grad += -1.0 * out->grad;
            };
        return out;
    }

    inline V mul(const V& a, const V& b) {
        auto out = Value::make(a->data * b->data);
        out->parents = { a, b };

        // out = a * b
        // dL/da += dL/dout * b
        // dL/db += dL/dout * a
        out->backward_fn = [out, a, b]() {
            a->grad += b->data * out->grad;
            b->grad += a->data * out->grad;
            };
        return out;
    }

    inline V div(const V& a, const V& b) {
        auto out = Value::make(a->data / b->data);
        out->parents = { a, b };

        // out = a / b
        // d(out)/da = 1/b
        // d(out)/db = -a/(b^2)
        out->backward_fn = [out, a, b]() {
            a->grad += (1.0 / b->data) * out->grad;
            b->grad += (-a->data / (b->data * b->data)) * out->grad;
            };
        return out;
    }

    inline V relu(const V& x) {
        auto out = Value::make(x->data > 0.0 ? x->data : 0.0);
        out->parents = { x };

        // out = max(0, x)
        // dout/dx = 1 if x>0, else 0
        out->backward_fn = [out, x]() {
            double mask = (x->data > 0.0) ? 1.0 : 0.0;
            x->grad += mask * out->grad;
            };
        return out;
    }

    inline V exp(const V& x) {
        auto out = Value::make(std::exp(x->data));
        out->parents = { x };

        // out = exp(x)
        // dout/dx = exp(x) = out
        out->backward_fn = [out, x]() {
            x->grad += out->data * out->grad;
            };
        return out;
    }

    inline V log(const V& x) {
        if (x->data <= 0.0) throw std::runtime_error("log(): x must be > 0");
        auto out = Value::make(std::log(x->data));
        out->parents = { x };

        // out = log(x)
        // dout/dx = 1/x
        out->backward_fn = [out, x]() {
            x->grad += (1.0 / x->data) * out->grad;
            };
        return out;
    }

    // useful operators
    inline V operator+(const V& a, const V& b) { return add(a, b); }
    inline V operator-(const V& a, const V& b) { return sub(a, b); }
    inline V operator*(const V& a, const V& b) { return mul(a, b); }
    inline V operator/(const V& a, const V& b) { return div(a, b); }

} // namespace ml::autograd
