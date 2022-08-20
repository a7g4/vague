#pragma once

#include "vague/estimate.hpp"
#include "vague/unscented_transform.hpp"
#include "vague/utility.hpp"
#include "Eigen/LU"

#include <chrono>

namespace vague {
    
template <typename StateSpaceT, typename ObservationSpaceT, typename ScalarT>
struct PredictedObservation {
    using StateSpace = StateSpaceT;
    using ObservationSpace = ObservationSpaceT;
    using Scalar = ScalarT;

    MeanAndCovariance<ObservationSpace, Scalar> observation;
    
    // TODO: Make this optional? So we don't need to calculate it unless its actually used for an update
    //       Will probably need to be to a variant so that the original state can be copied in?
    Eigen::Matrix<Scalar, StateSpace::N, ObservationSpace::N> cross_covariance;
};

template <typename StateSpaceT, typename DynamicsT, typename ScalarT, typename TimePointT>
class StateEstimator {
public:
    using StateSpace = StateSpaceT;
    using Dynamics = DynamicsT;
    using Scalar = ScalarT;
    using TimePoint = TimePointT;

    StateEstimator(const TimePoint& initial_time,
                   const MeanAndCovariance<StateSpace, Scalar>& initial_estimate,
                   Dynamics&& dynamics) : time(initial_time),
                                          estimate(initial_estimate),
                                          dynamics(std::move(dynamics)),
                                          process_noise([initial_estimate] (Scalar dt) { return dt * initial_estimate.covariance;}) {
        
    }

    void predict(TimePoint t) {
        const auto duration = (t-time);
        
        if (duration.count() == 0) { return; }
        if (duration.count() < 0) { throw std::runtime_error("Unable to wind back time"); }

        time = t;
        Scalar dt = std::chrono::duration_cast<std::chrono::duration<Scalar>>(duration).count();

        if constexpr (vague::utility::FunctionTypeCallableWith<decltype(estimate), decltype(dt)>::value) {
            // Path for linear & differentiable dynamics
            estimate = dynamics(estimate, dt).statistics();
        } else {
            // Need to sample sigma points for non-linear & non-differentiable dynamics
            estimate = dynamics(sample(estimate, unscented_transform::CubatureSigmaPoints()), dt).statistics();
        }
        estimate.covariance += process_noise(dt);
    }
    
    template <typename Observer, typename ... AugmentedState>
    PredictedObservation<StateSpace, typename Observer::To, Scalar> predicted_observation(const Observer& observer, const AugmentedState&... augmented_state) const {
        if constexpr (vague::utility::FunctionTypeCallableWith<decltype(estimate), AugmentedState...>::value) {
            throw std::runtime_error("Not yet implemented");
            return {
                observer(estimate, augmented_state...),
                estimate.covariance * observer.jacobian(estimate, augmented_state...)
            };
        } else {
            const auto sigma_points = sample(estimate, unscented_transform::CubatureSigmaPoints());
            const auto& [state_mean, state_centered_samples] = sigma_points.mean_centered_samples();

            const auto predicted_obs_sigma_points = observer(sigma_points, augmented_state...);
            const auto& [predicted_obs_mean, predicted_obs_centered_samples] = predicted_obs_sigma_points.mean_centered_samples();
            return {
                predicted_obs_sigma_points.statistics(),
                state_centered_samples * sigma_points.weights.asDiagonal() * predicted_obs_centered_samples.transpose()
            };
        }
    }
    
    template <typename ObservationSpace>
    void assimilate(const PredictedObservation<StateSpace, ObservationSpace, Scalar>& predicted_observation, const MeanAndCovariance<ObservationSpace, Scalar>& observation) {
        const Eigen::Matrix<Scalar, ObservationSpace::N, ObservationSpace::N> observation_covariance_sum = predicted_observation.observation.covariance + observation.covariance;
        const auto K = predicted_observation.cross_covariance * observation_covariance_sum.inverse();
        estimate.mean += K * (observation.mean - predicted_observation.observation.mean);
        estimate.covariance -= K * observation_covariance_sum * K.transpose();
    }

    TimePoint time;
    MeanAndCovariance<StateSpace, Scalar> estimate;
    Dynamics dynamics;
    // TODO: This feels clunky, do something nicer for process noise
    std::function<Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>(Scalar)> process_noise;
};

template <typename TimePoint, typename Estimate, typename Dynamics>
StateEstimator(TimePoint, Estimate, Dynamics) ->
    StateEstimator<typename Estimate::StateSpace,
                   Dynamics,
                   typename Estimate::Scalar,
                   TimePoint>;
}