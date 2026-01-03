#pragma once
#include <stdexcept>
#include <string>
#include <sstream>

namespace ml {

    inline [[noreturn]] void fail(const std::string& msg,
        const char* file,
        int line) {
        std::ostringstream oss;
        oss << "ML error: " << msg << " (" << file << ":" << line << ")";
        throw std::runtime_error(oss.str());
    }

}

#define ML_CHECK(cond, msg) \
    do { \
        if (!(cond)) ::ml::fail((msg), __FILE__, __LINE__); \
    } while (0)

#define ML_CHECK_EQ(a, b, msg) \
    do { \
        if (!((a) == (b))) ::ml::fail((msg), __FILE__, __LINE__); \
    } while (0)

#define ML_CHECK_LT(a, b, msg) \
    do { \
        if (!((a) < (b))) ::ml::fail((msg), __FILE__, __LINE__); \
    } while (0)