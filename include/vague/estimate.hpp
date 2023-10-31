#pragma once

#include "Eigen/Core"

namespace vague {

template <typename StateSpaceT, typename ScalarT>
struct Mean {
    using StateSpace = StateSpaceT;
    using Scalar = ScalarT;
    using Vector = Eigen::Matrix<Scalar, StateSpace::N, 1>;
    Mean(const Vector& mean_) noexcept : mean(mean_) { }
    Vector mean;
};

template <typename StateSpaceT, typename ScalarT>
struct MeanAndCovariance : Mean<StateSpaceT, ScalarT> {
    using StateSpace = StateSpaceT;
    using Scalar = ScalarT;
    using Vector = Eigen::Matrix<Scalar, StateSpace::N, 1>;
    using Matrix = Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>;
    MeanAndCovariance(const Vector& mean, const Matrix& covariance_) noexcept :
        Mean<StateSpace, Scalar>(mean),
        covariance(covariance_) { }
    Matrix covariance;
};

struct UniformWeightsTag { };

template <typename StateSpaceT, typename ScalarT, int N_SamplesT>
struct WeightedSamples {
    constexpr static int NSamples = N_SamplesT;
    using StateSpace = StateSpaceT;
    using Scalar = ScalarT;
    using Vector = Eigen::Matrix<Scalar, StateSpace::N, 1>;
    using Matrix = Eigen::Matrix<Scalar, StateSpace::N, StateSpace::N>;
    using WeightsVector = Eigen::Matrix<Scalar, NSamples, 1>;
    using SamplesMatrix = Eigen::Matrix<Scalar, StateSpace::N, NSamples>;

    WeightedSamples(const SamplesMatrix& samples_, const WeightsVector& weights_) noexcept :
        samples(samples_),
        weights(weights_) { }
    WeightedSamples(SamplesMatrix&& samples_, WeightsVector&& weights_) noexcept :
        samples(std::move(samples_)),
        weights(std::move(weights)) { }

    WeightedSamples(const SamplesMatrix& samples_, UniformWeightsTag /*unused*/) noexcept :
        samples(samples_),
        weights(WeightsVector::Constant(1. / NSamples)) { }
    WeightedSamples(SamplesMatrix&& samples_, UniformWeightsTag /*unused*/) noexcept :
        samples(std::move(samples_)),
        weights(WeightsVector::Constant(1. / NSamples)) { }

    WeightedSamples(const WeightedSamples& copy) noexcept = default;
    WeightedSamples(WeightedSamples&& move) noexcept = default;
    ~WeightedSamples() noexcept = default;

    WeightedSamples& operator=(const WeightedSamples& copy) = default;
    WeightedSamples& operator=(WeightedSamples&& move) noexcept = default;

    auto operator[](const size_t& index) const noexcept { return samples.col(index); }

    constexpr static size_t AngleExpandedSize = StateSpace::N + StateSpace::ANGLES.size();
    using AngleExpandedSamples = Eigen::Matrix<Scalar, AngleExpandedSize, NSamples>;
    using AngleExpandedVector = Eigen::Matrix<Scalar, AngleExpandedSize, 1>;
    using AngleExpandedMatrix = Eigen::Matrix<Scalar, AngleExpandedSize, AngleExpandedSize>;
    AngleExpandedSamples expand_angles() const noexcept {
        if constexpr (StateSpace::ANGLES.size() == 0) { return samples; }
        // To calculate reasonable statistics around angles:
        //  - First each angle to cartesian coordinates (ie. expand theta into cos(theta), sin(theta))
        //  - Average (element-wise) these "expanded" elements
        //  - Collapse back into polar coordinates
        using AngleExpandedWeightedSamples = Eigen::Matrix<Scalar, AngleExpandedSize, NSamples>;

        AngleExpandedWeightedSamples expanded_samples;
        expanded_samples.template topRows<StateSpace::N>() = samples;

        // To minimise copying, the expanded vector will add 1 element for each angle component.
        // The original angle index will be used for the cosine component and the extra element will store the sin
        // component.
        //
        // Eg. State Space             {x, y,     theta,  z,     phi}
        //     will get expanded into: {x, y, cos(theta), z, cos(phi), sin(theta), sin(phi)}
        //     where theta and phi are angles.
        for (size_t i = 0; i < StateSpace::ANGLES.size(); i++) {
            expanded_samples.row(StateSpace::ANGLES[i]) = samples.row(StateSpace::ANGLES[i]).array().cos();
            expanded_samples.row(StateSpace::N + i) = samples.row(StateSpace::ANGLES[i]).array().sin();
        }
        return expanded_samples;
    }

    MeanAndCovariance<StateSpace, Scalar> statistics() const noexcept {
        const auto& [mean, centered] = mean_centered_samples();
        Matrix covariance = centered * weights.asDiagonal() * centered.transpose();
        return {mean, covariance};
    }

    struct MeanCenteredSamples {
        Vector mean;
        Eigen::Matrix<Scalar, StateSpace::N, NSamples> samples;
    };
    MeanCenteredSamples mean_centered_samples() const noexcept {
        if constexpr (StateSpace::ANGLES.size() == 0) {
            Vector mean = samples * weights;
            SamplesMatrix centered = samples.colwise() - mean;
            return {mean, centered};
        } else {
            AngleExpandedSamples expanded_samples = expand_angles();
            AngleExpandedVector expanded_mean = expanded_samples * weights;
            Vector mean = expanded_mean.template topRows<StateSpace::N>();
            for (size_t i = 0; i < StateSpace::ANGLES.size(); i++) {
                mean[StateSpace::ANGLES[i]] = std::atan2(expanded_mean[StateSpace::N + i],      // sin component
                                                         expanded_mean[StateSpace::ANGLES[i]]); // cos component
            }

            SamplesMatrix centered = samples.colwise() - mean;
            // Unwrap each angle - Note: This assumes deviations are meant to be within some small angle of the mean
            // TODO: Maybe make this behaviour a parameter or something?
            centered(StateSpace::ANGLES, Eigen::all) =
                centered(StateSpace::ANGLES, Eigen::all).unaryExpr([](const Scalar& angle_diff) {
                    static constexpr Scalar PI = 3.141592653589793238462643383279;
                    return std::fmod(angle_diff, PI);
                });
            return {mean, centered};
        }
    }

    Eigen::Matrix<Scalar, StateSpace::N, NSamples> samples;
    WeightsVector weights;
};

} // namespace vague