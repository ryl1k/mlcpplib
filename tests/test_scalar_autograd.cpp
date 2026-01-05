#include <cassert>
#include <cmath>
#include <iostream>
#include "ml/autograd/value.hpp"

int main() {
    using namespace ml::autograd;

    // ====== Test 1: y = x*x + x, x=3 => dy/dx = 2x+1 = 7 ======
    auto x = Value::make(3.0);
    auto y = x * x;      // x^2
    auto z = y + x;      // x^2 + x

    z->backward();

    std::cout << "z = " << z->data << "\n";      // 12
    std::cout << "dz/dx = " << x->grad << "\n";  // 7

    assert(std::abs(z->data - 12.0) < 1e-12);
    assert(std::abs(x->grad - 7.0) < 1e-12);

    // ====== Test 2: N-Nodes: L = (x*2) + (x*3) => dL/dx = 5 ======
    auto x2 = Value::make(10.0);
    auto two = Value::make(2.0);
    auto three = Value::make(3.0);

    auto b = x2 * two;    // 2x
    auto c = x2 * three;  // 3x
    auto L = b + c;       // 5x

    L->backward();

    std::cout << "dL/dx = " << x2->grad << "\n"; // 5
    assert(std::abs(x2->grad - 5.0) < 1e-12);

    // ====== Тест 3: log(x^2) chain rule: f=log(x*x), x=4 => df/dx = 2/x = 0.5 ======
    auto x3 = Value::make(4.0);
    auto f = log(x3 * x3);
    f->backward();
    std::cout << "df/dx = " << x3->grad << "\n"; // 0.5
    assert(std::abs(x3->grad - 0.5) < 1e-12);

    std::cout << "All scalar autograd tests passed ✅\n";
    return 0;
}
