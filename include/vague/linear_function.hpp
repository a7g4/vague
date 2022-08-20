#pragma once

#include <Eigen/Core>
#include "vague/utility.hpp"

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

    LinearFunction(const Matrix& f) : F(f) { }
    LinearFunction(Matrix&& f) : F(std::move(f)) { }

    template <typename ... AugmentedState>
    Output operator()(const Input& input, const AugmentedState&... augmented_state) const {
        // TODO: Should we even have augmented state for a linear function?
        return F * input;
    }

    template <typename ... AugmentedState>
    JacobianOutput jacobian(const Input& input, const AugmentedState&... augmented_state) const {
        return F;
    }

    template <typename ... AugmentedState>
    Mean<To, Scalar> operator()(const Mean<From, Scalar>& input, const AugmentedState&... augmented_state) const {
        // TODO: Should we even have augmented state for a linear function?
        return Mean<To, Scalar>(F * input.mean);
    }

    template <typename ... AugmentedState>
    MeanAndCovariance<To, Scalar> operator()(const MeanAndCovariance<From, Scalar>& input, const AugmentedState&... augmented_state) const {
        // TODO: Should we even have augmented state for a linear function?
        return MeanAndCovariance<To, Scalar>(F * input.mean, F * input.covariance * F.transpose());
    }

    template <int N_Points, typename ... AugmentedState>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input, const AugmentedState&... augmented_state) const {
        // TODO: Should we even have augmented state for a linear function?
        return WeightedSamples<To, Scalar, N_Points>(F * input.samples, input.weights);
    }

    Matrix F;
};

}