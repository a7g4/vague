#include "catch.hpp"
#include "vague/state_spaces.hpp"
#include "vague/estimate.hpp"
#include <iostream>

#include "vague/differentiable_function.hpp"
#include "test_helpers.hpp"

TEST_CASE("Truncation projection, raw Eigen matrix", "[differentiable_function]" ) {
    using From = vague::state_spaces::CartesianPosYaw2D;
    using To = vague::state_spaces::CartesianPos2D;

    vague::DifferentiableFunction f(From(), To(),
                                    [](const Eigen::Vector3d i){ return Eigen::Matrix<double, 2, 1>(i.topRows(2));},
                                    [](const Eigen::Vector3d){ return Eigen::Matrix<double, 2, 3>({{1, 0, 0},{0, 1, 0}});});
    
    Eigen::Vector3d input(1,2,3);
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(1,2), f(input));
}

TEST_CASE("Truncation projection, Mean", "[differentiable_function]" ) {
    using From = vague::state_spaces::CartesianPosYaw2D;
    using To = vague::state_spaces::CartesianPos2D;

    vague::DifferentiableFunction f(From(), To(),
                                    [](const Eigen::Vector3d i){ return Eigen::Matrix<double, 2, 1>(i.topRows(2));},
                                    [](const Eigen::Vector3d){ return Eigen::Matrix<double, 2, 3>({{1, 0, 0},{0, 1, 0}});});
    
    vague::Mean<From, double> mean({1,2,3});
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(1,2), f(mean).mean);
}

TEST_CASE("Truncation projection, MeanAndCovariance", "[differentiable_function]" ) {
    using From = vague::state_spaces::CartesianPosYaw2D;
    using To = vague::state_spaces::CartesianPos2D;

    vague::DifferentiableFunction f(From(), To(),
                                    [](const Eigen::Vector3d i){ return Eigen::Matrix<double, 2, 1>(i.topRows(2));},
                                    [](const Eigen::Vector3d){ return Eigen::Matrix<double, 2, 3>({{1, 0, 0},{0, 1, 0}});});
    
    vague::MeanAndCovariance<From, double> mac({1,2,3}, Eigen::Matrix3d({{1,0,0},{0,2,0},{0,3,3}}));
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(1,2), f(mac).mean);
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Matrix2d({{1, 0}, {0, 2}}), f(mac).covariance);
}

TEST_CASE("Truncation projection, WeightedSamples", "[differentiable_function]" ) {
    using From = vague::state_spaces::CartesianPosYaw2D;
    using To = vague::state_spaces::CartesianPos2D;

    vague::DifferentiableFunction f(From(), To(),
                                    [](const Eigen::Vector3d i){ return Eigen::Matrix<double, 2, 1>(i.topRows(2));},
                                    [](const Eigen::Vector3d){ return Eigen::Matrix<double, 2, 3>({{1, 0, 0},{0, 1, 0}});});
    
    vague::WeightedSamples<From, double, 3> sps(Eigen::Matrix3d({{1,2,3},{3,3,3},{0,4,8}}), {0.3, 0.3, 0.3});

    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(1,3), f(sps)[0]);
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(2,3), f(sps)[1]);
    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector2d(3,3), f(sps)[2]);

    CHECK_MATRIX_NEARLY_EQUAL(Eigen::Vector3d(0.3, 0.3, 0.3), f(sps).weights);
}