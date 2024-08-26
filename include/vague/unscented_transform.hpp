#pragma once

#include "Eigen/Cholesky"
#include "vague/estimate.hpp"

#include <iostream>

namespace vague {

namespace unscented_transform {
struct CubatureSigmaPoints { };
struct JulierSigmaPoints {
    float kappa = 0;
};
} // namespace unscented_transform

template <typename StateSpace, typename Scalar>
WeightedSamples<StateSpace, Scalar, StateSpace::N * 2> sample(MeanAndCovariance<StateSpace, Scalar> distribution,
                                                              unscented_transform::CubatureSigmaPoints /*unused*/) noexcept {
    Eigen::LLT<Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>> llt_solver(distribution.covariance);
    Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N> sqrt_sigma;
    if (llt_solver.info() != Eigen::Success) {
        // TODO: Emit a warning somehow that LDLT was needed, this is a performance hit
        Eigen::LDLT<Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>> ldlt_solver(distribution.covariance);

        if (ldlt_solver.info() != Eigen::Success) {
            throw std::runtime_error("LLT and LDLT solvers failed on the covariance matrix");
        }

        sqrt_sigma = ldlt_solver.matrixL();

        // The cwiseMin(0) is to protect against cases where D has negative elements. That means the covariance wasn't
        // actually positive semi-definite but this should mean that sqrt_sigma * sqrt_sigmaT is a close approximation
        // of the input covariance.
        sqrt_sigma = sqrt_sigma * ldlt_solver.vectorD().cwiseMax(0).cwiseSqrt().asDiagonal();
    } else {
        sqrt_sigma = llt_solver.matrixL();
    }

    constexpr int NPoints = StateSpace::N * 2;
    Eigen::Matrix<Scalar, StateSpace::N, NPoints> sigma_points;
    Eigen::Matrix<Scalar, NPoints, 1> weights = Eigen::Matrix<Scalar, NPoints, 1>::Constant(1. / (2 * StateSpace::N));

    Scalar scale = std::sqrt(static_cast<Scalar>(StateSpace::N));
    sigma_points(Eigen::all, Eigen::seq(0, Eigen::last, 2)) = (scale * sqrt_sigma).colwise() + distribution.mean;
    sigma_points(Eigen::all, Eigen::seq(1, Eigen::last, 2)) = (-scale * sqrt_sigma).colwise() + distribution.mean;

    return {sigma_points, weights};
}

} // namespace vague
