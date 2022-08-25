#include "catch.hpp"
#include "test_helpers.hpp"
#include "vague/estimate.hpp"

#include <iostream>

namespace {
constexpr double PI = 3.141592653589793238462643383279;
constexpr double EPSILON = 1e-10;

struct SixDofWithWeirdOrder {
    enum Elements { X, Yaw, Y, Pitch, Z, Roll, N };
    constexpr static std::array<size_t, 3> ANGLES {Roll, Yaw, Pitch};
};
} // namespace

TEST_CASE("Test weighted samples wrapping around zero/two pi", "[angle_statistics]") {
    Eigen::Matrix<double, 6, 3> data;
    // clang-format off
    data.col(0) << 1,          0.1, 1,          PI / 4, 3, 3 * PI / 4;
    data.col(1) << 2, 2 * PI - 0.1, 1, 2 * PI - PI / 4, 2, 5 * PI / 4;
    data.col(2) << 3,            0, 1,          4 * PI, 1, 4 * PI / 4;
    // clang-format on

    vague::WeightedSamples<SixDofWithWeirdOrder, double, 3> samples(data, {1. / 3, 1. / 3, 1. / 3});

    auto statistics = samples.statistics();
    std::cout << statistics.mean.transpose() << std::endl;
    CHECK_THAT(statistics.mean[0], Catch::Matchers::WithinAbs(2., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[1], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    CHECK_THAT(statistics.mean[2], Catch::Matchers::WithinAbs(1., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[3], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    CHECK_THAT(statistics.mean[4], Catch::Matchers::WithinAbs(2., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[5], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    // TODO: Check covariance
}

TEST_CASE("Test weighted samples - weighting works", "[angle_statistics]") {
    Eigen::Matrix<double, 6, 6> data;
    // clang-format off
    data.col(0) << 1,          0.1, 1,          PI / 4, 3, 3 * PI / 4;
    data.col(1) << 2, 2 * PI - 0.1, 1, 2 * PI - PI / 4, 2, 5 * PI / 4;
    data.col(2) << 3,            0, 1,          4 * PI, 1, 4 * PI / 4;
    data.col(3) << 4,            9, 2, 6 * PI - PI / 4, 0, 6 * PI / 4;
    data.col(4) << 5,            9, 2, 8 * PI - PI / 4, 1, 9 * PI / 4;
    data.col(5) << 6,            0, 2, 0 * PI - PI / 4, 2, 7 * PI / 4;
    // clang-format on

    vague::WeightedSamples<SixDofWithWeirdOrder, double, 6> samples(data, {1. / 3, 1. / 3, 1. / 3, 0, 0, 0});

    auto statistics = samples.statistics();
    std::cout << statistics.mean.transpose() << std::endl;
    CHECK_THAT(statistics.mean[0], Catch::Matchers::WithinAbs(2., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[1], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    CHECK_THAT(statistics.mean[2], Catch::Matchers::WithinAbs(1., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[3], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    CHECK_THAT(statistics.mean[4], Catch::Matchers::WithinAbs(2., EPSILON));
    CHECK_THAT(std::fmod(statistics.mean[5], PI), Catch::Matchers::WithinAbs(0., EPSILON));
    // TODO: Check covariance
}