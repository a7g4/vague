#include "catch.hpp"
#include "vague/state_estimator.hpp"
#include "vague/arbitrary_function.hpp"
#include "vague/linear_function.hpp"
#include "vague/state_spaces.hpp"

#include "test_helpers.hpp"

TEST_CASE("Simple test, arbitrary dynamics, arbitrary observation", "[state_estimator]" ) {
    using StateSpace = vague::state_spaces::CartesianPosVel2D;
    vague::MeanAndCovariance<StateSpace, double> initialEstimate(
        {0.0, 0.0, 0.0, 0.0},
        Eigen::Matrix<double, 4, 4>::Identity()
    );
    vague::ArbitraryFunction dynamics(StateSpace(), StateSpace(), [](const Eigen::Matrix<double, 4, 1> state, double dt){
            Eigen::Matrix<double, 4, 1> newState = state;
            newState[StateSpace::X] += dt * state[StateSpace::V_X];
            newState[StateSpace::Y] += dt * state[StateSpace::V_Y];
            return newState;
    });
    vague::ArbitraryFunction simple_projection(vague::state_spaces::CartesianPos2D(), StateSpace(), [](const Eigen::Matrix<double, 4, 1> state) {
            Eigen::Matrix<double, 2, 1> projection = state.topRows(2);
            return projection;
    });
    auto initial_time = std::chrono::system_clock::time_point();
    vague::StateEstimator estimator(initial_time, initialEstimate, std::move(dynamics));
    
    for (size_t i = 0; i < 10; i++) {
        const auto time = initial_time + std::chrono::seconds(1 * i);
        estimator.predict(time + std::chrono::seconds(1));
        const auto& predicted_obs = estimator.predicted_observation(simple_projection);
        
        vague::MeanAndCovariance<vague::state_spaces::CartesianPos2D, double> observation(
            Eigen::Vector2d(i, i * 5.0),
            Eigen::Matrix2d::Identity()
        );
        estimator.assimilate(predicted_obs, observation);        
        std::cout << estimator.estimate.mean.transpose() << "\n=======\n";
        std::cout << estimator.estimate.covariance << "\n=======\n";
    }
}

TEST_CASE("Simple test, arbitrary dynamics, linear observations", "[state_estimator]" ) {
    using StateSpace = vague::state_spaces::CartesianPosVel2D;
    vague::MeanAndCovariance<StateSpace, double> initialEstimate(
        {0.0, 0.0, 0.0, 0.0},
        Eigen::Matrix<double, 4, 4>::Identity()
    );
    vague::ArbitraryFunction dynamics(StateSpace(), StateSpace(), [](const Eigen::Matrix<double, 4, 1> state, double dt){
            Eigen::Matrix<double, 4, 1> newState = state;
            newState[StateSpace::X] += dt * state[StateSpace::V_X];
            newState[StateSpace::Y] += dt * state[StateSpace::V_Y];
            return newState;
    });
    vague::LinearFunction<vague::state_spaces::CartesianPos2D, vague::state_spaces::CartesianPosVel2D, double> simple_projection(
            Eigen::Matrix<double, 2, 4> {{1, 0, 0, 0}, {0, 1, 0, 0}}
    );
    auto initial_time = std::chrono::system_clock::time_point();
    vague::StateEstimator estimator(initial_time, initialEstimate, std::move(dynamics));
    
    for (size_t i = 0; i < 10; i++) {
        const auto time = initial_time + std::chrono::seconds(1 * i);
        estimator.predict(time + std::chrono::seconds(1));
        const auto& predicted_obs = estimator.predicted_observation(simple_projection);
        
        vague::MeanAndCovariance<vague::state_spaces::CartesianPos2D, double> observation(
            Eigen::Vector2d(i, i * 5.0),
            Eigen::Matrix2d::Identity()
        );
        estimator.assimilate(predicted_obs, observation);        
        std::cout << estimator.estimate.mean.transpose() << "\n=======\n";
        std::cout << estimator.estimate.covariance << "\n=======\n";
    }
}