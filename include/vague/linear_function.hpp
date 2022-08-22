#pragma once

#include <Eigen/Core>
#include "vague/utility.hpp"
#include <utility>

namespace vague {

template <typename ToT, typename FromT, typename ScalarT>
struct LinearFunction {
    
    using To = ToT;
    using From = FromT;
    using Scalar = ScalarT;
    constexpr static size_t DIM_RANGE = To::N;
    constexpr static size_t DIM_DOMAIN = From::N;
    using Matrix = Eigen::Matrix<Scalar, DIM_RANGE, DIM_DOMAIN>;
    using Input = Eigen::Matrix<Scalar, DIM_DOMAIN, 1>;
    using Output = Eigen::Matrix<Scalar, DIM_RANGE, 1>;
    using JacobianOutput = Eigen::Matrix<Scalar, DIM_RANGE, DIM_DOMAIN>;

    LinearFunction(const Matrix& f) noexcept : F(f) { }
    LinearFunction(Matrix&& f) noexcept : F(std::move(f)) { }

    Output operator()(const Input& input) const noexcept {
        return F * input;
    }

    JacobianOutput jacobian(const Input& input) const noexcept {
        return F;
    }

    Mean<To, Scalar> operator()(const Mean<From, Scalar>& input) const noexcept {
        return Mean<To, Scalar>(F * input.mean);
    }

    MeanAndCovariance<To, Scalar> operator()(const MeanAndCovariance<From, Scalar>& input) const noexcept {
        return MeanAndCovariance<To, Scalar>(F * input.mean, F * input.covariance * F.transpose());
    }

    template <int N_Points>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input) const noexcept {
        return WeightedSamples<To, Scalar, N_Points>(F * input.samples, input.weights);
    }

    Matrix F;
};

}