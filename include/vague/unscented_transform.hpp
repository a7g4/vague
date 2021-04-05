#pragma once

#include "vague/estimate.hpp"
#include "Eigen/Cholesky"
#include <iostream>

namespace vague {
    
namespace unscented_transform {
struct CubatureSigmaPoints { };
struct JulierSigmaPoints { float kappa = 0; };
}


template <typename StateSpace, typename Scalar>
WeightedSamples<StateSpace, Scalar, StateSpace::N * 2> sample(MeanAndCovariance<StateSpace, Scalar> distribution,
                                                              unscented_transform::CubatureSigmaPoints) {
    Eigen::LLT<Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>> llt_solver(distribution.covariance);
    if (llt_solver.info() != Eigen::Success) {
        // TODO: Fall back to LDLT solver?
        throw std::runtime_error("Unable to perform Cholesky decomposition on covariance");
    }
    Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N> sqrt_sigma = llt_solver.matrixL();
    
    constexpr int N_Points = StateSpace::N * 2;
    Eigen::Matrix<Scalar, StateSpace::N, N_Points> sigma_points;
    Eigen::Matrix<Scalar, N_Points, 1> weights = Eigen::Matrix<Scalar, N_Points, 1>::Constant(1./(2*StateSpace::N));
    
    Scalar scale = std::sqrt(static_cast<Scalar>(StateSpace::N));
    sigma_points(Eigen::all, Eigen::seq(0,Eigen::last,2)) = (scale * sqrt_sigma).colwise() + distribution.mean;
    sigma_points(Eigen::all, Eigen::seq(1,Eigen::last,2)) = (-scale * sqrt_sigma).colwise() + distribution.mean;
    
    return {sigma_points, weights};
}

template <typename Space1, typename Space2, typename Scalar, int N_Points>
Eigen::Matrix<Scalar, Space1::N, Space2::N> cross_covariance() {

}

}