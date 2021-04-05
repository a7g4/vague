#include "catch.hpp"
#include "vague/estimate.hpp"

#include "test_helpers.hpp"

struct SixDofWithWeirdOrder {
    enum Elements { X, Yaw, Y, Pitch, Z, Roll, N };
    constexpr static std::array<size_t, 3> Angles { Roll, Yaw, Pitch };
};

TEST_CASE("Test weighted samples wrapping around zero/two pi", "[angle_statistics]" ) {
    constexpr double pi = std::acos(1);
    Eigen::Matrix<double, 6, 3> data;
    data.col(0) << 1,        0.1, 1,        pi/4, 3, 3*pi/4;
    data.col(1) << 2, 2*pi - 0.1, 1, 2*pi - pi/4, 2, 5*pi/4;
    data.col(2) << 3,          0, 1, 4*pi - pi/4, 1, 4*pi/4;

    vague::WeightedSamples<SixDofWithWeirdOrder, double, 3> samples(data, {1./3, 1./3, 1./3});
    
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Matrix<double,6,1>(2,0,1,0,2,0), samples.statistics().mean, 0.001);
    // TODO: Check covariance
}

TEST_CASE("Test weighted samples - weighting works", "[angle_statistics]" ) {
    constexpr double pi = std::acos(1);
    Eigen::Matrix<double, 6, 6> data;
    data.col(0) << 1,        0.1, 1,        pi/4, 3, 3*pi/4;
    data.col(1) << 2, 2*pi - 0.1, 1, 2*pi - pi/4, 2, 5*pi/4;
    data.col(2) << 3,          0, 1, 4*pi - pi/4, 1, 4*pi/4;
    data.col(3) << 4,          9, 2, 6*pi - pi/4, 0, 6*pi/4;
    data.col(4) << 5,          9, 2, 8*pi - pi/4, 1, 9*pi/4;
    data.col(5) << 6,          0, 2, 0*pi - pi/4, 2, 7*pi/4;

    vague::WeightedSamples<SixDofWithWeirdOrder, double, 6> samples(data, {1./3, 1./3, 1./3, 0, 0, 0});
    
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Matrix<double,6,1>(2,0,1,0,2,0), samples.statistics().mean, 0.001);
    // TODO: Check covariance
}