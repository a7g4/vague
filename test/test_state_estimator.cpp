#include "catch.hpp"
#include "test_helpers.hpp"
#include "vague/arbitrary_function.hpp"
#include "vague/linear_function.hpp"
#include "vague/state_estimator.hpp"
#include "vague/state_spaces.hpp"

TEST_CASE("Simple test, arbitrary dynamics, arbitrary observation", "[state_estimator]") {
    using StateSpace = vague::state_spaces::CartesianPosVel2D;
    vague::MeanAndCovariance<StateSpace, double> initial_estimate({0.0, 0.0, 0.0, 0.0},
                                                                  Eigen::Matrix<double, 4, 4>::Identity());
    vague::ArbitraryFunction dynamics(StateSpace(), StateSpace(), [](const Eigen::Matrix<double, 4, 1>& state, double dt) {
        Eigen::Matrix<double, 4, 1> new_state = state;
        new_state[StateSpace::X] += dt * state[StateSpace::V_X];
        new_state[StateSpace::Y] += dt * state[StateSpace::V_Y];
        return new_state;
    });
    vague::TimeDependentAdditiveProcessNoise<StateSpace, double> process_noise(initial_estimate.covariance);
    vague::ArbitraryFunction simple_projection(
        vague::state_spaces::CartesianPos2D(), StateSpace(), [](const Eigen::Matrix<double, 4, 1>& state) {
            Eigen::Matrix<double, 2, 1> projection = state.topRows(2);
            return projection;
        });
    auto initial_time = std::chrono::system_clock::time_point();
    vague::StateEstimator estimator(initial_time, initial_estimate);

    for (size_t i = 0; i < 10; i++) {
        const auto time = initial_time + std::chrono::seconds(1 * i);
        estimator.predict(time + std::chrono::seconds(1), dynamics, process_noise);
        const auto& predicted_obs = estimator.predict_observation(simple_projection);

        vague::MeanAndCovariance<vague::state_spaces::CartesianPos2D, double> observation(Eigen::Vector2d(i, i * 5.0),
                                                                                          Eigen::Matrix2d::Identity());
        estimator.assimilate(predicted_obs, observation);
        std::cout << estimator.estimate.mean.transpose() << "\n=======\n";
        std::cout << estimator.estimate.covariance << "\n=======\n";
    }
}

TEST_CASE("Simple test, arbitrary dynamics, linear observations", "[state_estimator]") {
    using StateSpace = vague::state_spaces::CartesianPosVel2D;
    vague::MeanAndCovariance<StateSpace, double> initial_estimate({0.0, 0.0, 0.0, 0.0},
                                                                  Eigen::Matrix<double, 4, 4>::Identity());
    vague::ArbitraryFunction dynamics(StateSpace(), StateSpace(), [](const Eigen::Matrix<double, 4, 1>& state, double dt) {
        Eigen::Matrix<double, 4, 1> new_state = state;
        new_state[StateSpace::X] += dt * state[StateSpace::V_X];
        new_state[StateSpace::Y] += dt * state[StateSpace::V_Y];
        return new_state;
    });
    vague::TimeDependentAdditiveProcessNoise<StateSpace, double> process_noise(initial_estimate.covariance);
    vague::LinearFunction<vague::state_spaces::CartesianPos2D, vague::state_spaces::CartesianPosVel2D, double> simple_projection(
        Eigen::Matrix<double, 2, 4> {
            {1, 0, 0, 0},
            {0, 1, 0, 0}
    });
    auto initial_time = std::chrono::system_clock::time_point();
    vague::StateEstimator estimator(initial_time, initial_estimate);

    for (size_t i = 0; i < 10; i++) {
        const auto time = initial_time + std::chrono::seconds(1 * i);
        estimator.predict(time + std::chrono::seconds(1), dynamics, process_noise);
        const auto& predicted_obs = estimator.predict_observation(simple_projection);

        vague::MeanAndCovariance<vague::state_spaces::CartesianPos2D, double> observation(Eigen::Vector2d(i, i * 5.0),
                                                                                          Eigen::Matrix2d::Identity());
        estimator.assimilate(predicted_obs, observation);
        std::cout << estimator.estimate.mean.transpose() << "\n=======\n";
        std::cout << estimator.estimate.covariance << "\n=======\n";
    }
}