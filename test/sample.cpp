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

TEST_CASE( "Flux capacitor charged", "[flux_capacitor]" ) {

    using Input = Eigen::Matrix<double, 1, 1>;
    using Output = Eigen::Matrix<double, 2, 1>;

    vague::LinearFunction<From, To, double> f(Eigen::Matrix<double, 2, 1> {{1}, {2}});

    vague::DifferentiableFunction<From, To, std::function<Output(const Input&)>, std::function<Output(const Input&)>> df(
        [](const Input& i) -> Output { return Output(i[0], i[0] * i[0]); },
        [](const Input& i) -> Output { return Output(1, 2 * i[0]); }
    );

    vague::ArbitraryFunction<From, To, std::function<Output(const Input&)>> af(
        [](const Input& i) -> Output { return Output(i[0], i[0] * i[0]); }
    );

    auto af2 = vague::ArbitraryFunction(
        From(),
        To(),
        [](const Input& i) -> Output { return Output (i[0], i[0] * i[0]); }
    );

    Input t {1.5};

    std::cout << t << "\n\n";
    std::cout << f(t) << "\n\n";
    std::cout << df(t) << "\n\n";

    // auto a = [](Input i, Input b) -> Output { return Output {i[0], i[0] * i[0]}; };
    // std::cout << std::is_same<vague::utility::FunctionType<decltype(a)>::Output, Output>::value;
    // std::cout << std::is_same<vague::utility::FunctionType<decltype(a)>::Input<0>, Input>::value;
    // std::cout << vague::utility::FunctionType<decltype(a)>::Inputs;
    // std::cout << vague::utility::FunctionType<decltype(a)>::Input<0>::RowsAtCompileTime;

    std::cout << "sizeof(f) = " << sizeof(f) << "\n";
    std::cout << "sizeof(df) = " << sizeof(df) << "\n";
    std::cout << "sizeof(df.F) = " << sizeof(df.F) << "\n";
    std::cout << "sizeof(df.J) = " << sizeof(df.J) << "\n";

    vague::state_spaces::CartesianPosYaw2D ss;
    std::cout << "sizeof(CartesianPosYaw2D) = " << sizeof(ss) << "\n\n";

    vague::WeightedSamples<vague::state_spaces::CartesianPos2D, double, 6> samples(
        Eigen::Matrix<double, 2, 6> {{1, 2, 3, 1, 2, 3}, {2, 2, 2, 3, 3, 3}},
        Eigen::Matrix<double, 6, 1>::Ones() / 6.
    );

    std::cout << samples[1] << std::endl;

    std::cout << "mean = \n" << samples.statistics().mean << std::endl;
    std::cout << "covariance = \n" << samples.statistics().covariance << "\n\n" << std::endl;

    vague::Mean<vague::state_spaces::CartesianPos2D, double> mean(Eigen::Matrix<double, 2, 1> {{1}, {2}});
    vague::MeanAndCovariance<vague::state_spaces::CartesianPos2D, double> mac(Eigen::Matrix<double, 2, 1> {{1}, {2}},
                                                                               Eigen::Matrix<double, 2, 2> {{1, 2}, {2, 3}});

    std::cout << mac.covariance << std::endl;
    std::cout << mac.mean << std::endl;

    std::cout << "sizeof(From) = " << sizeof(From) << std::endl;
    std::cout << "sizeof(Box2D) = " << sizeof(vague::state_spaces::Box2D) << std::endl;

    REQUIRE( 1 == 1 );
}