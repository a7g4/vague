#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "vague/state_spaces.hpp"
#include "vague/estimate.hpp"
#include <iostream>


#include "vague/arbitrary_function.hpp"
#include "vague/differentiable_function.hpp"
#include "vague/linear_function.hpp"

struct From {
    enum Elements { E1, N };
    constexpr static std::array<size_t, 0> Angles {};
};

struct To {
    enum Elements { E1, E2, N };
    constexpr static std::array<size_t, 0> Angles {};
};

TEST_CASE( "Benchmark different function types", "[benchmark]" ) {

    using Input = Eigen::Matrix<double, 1, 1>;
    using Output = Eigen::Matrix<double, 2, 1>;

    std::function<Output(const Input&)> fn = [](const Input& i) -> Output { return Output {i[0], i[0] * i[0]}; };

    vague::ArbitraryFunction<From, To, std::function<Output(const Input&)>> af_function(fn);

    auto af_function_auto = vague::ArbitraryFunction(From(), To(), fn);

    auto af_lambda = vague::ArbitraryFunction(
        From(),
        To(),
        [](const Input& i) -> Output { return Output {i[0], i[0] * i[0]}; }
    );

    Input t {1.5};

    BENCHMARK("Arbitrary Function (std::function)") {
        return af_function(t);
    };
    BENCHMARK("Arbitrary Function (std::function, auto)") {
        return af_function_auto(t);
    };
    BENCHMARK("Arbitrary Function (Lambda)") {
        return af_lambda(t);
    };

    REQUIRE(af_function(t) == af_lambda(t));
}