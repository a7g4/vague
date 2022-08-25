#pragma once

#include "Eigen/Core"
#include "vague/estimate.hpp"
#include "vague/utility.hpp"

#include <utility>

namespace vague {

template <typename ToT, typename FromT, typename FunctionT, typename JacobianT>
struct DifferentiableFunction {

    using To = ToT;
    using From = FromT;
    using Function = FunctionT;
    using Jacobian = JacobianT;

    constexpr static size_t DimRange = To::N;
    constexpr static size_t DimDomain = From::N;

    using Scalar = typename utility::FunctionType<Function>::Output::Scalar;
    using Input = Eigen::Matrix<Scalar, DimDomain, 1>;
    using Output = Eigen::Matrix<Scalar, DimRange, 1>;
    using JacobianOutput = Eigen::Matrix<Scalar, DimRange, DimDomain>;

    static_assert(std::is_same<Input, typename utility::FunctionType<Function>::template Input<0>>::value,
                  "First input to the function must be an Eigen vector representing the 'From' space");
    static_assert(std::is_same<Input, typename utility::FunctionType<Jacobian>::template Input<0>>::value,
                  "First input to the jacobian must be an Eigen vector representing the 'From' space");

    static_assert(std::is_same<Output, typename utility::FunctionType<Function>::Output>::value,
                  "Output of the function must be an Eigen vector representing the 'To' space");
    static_assert(std::is_same<JacobianOutput, typename utility::FunctionType<Jacobian>::Output>::value,
                  "Output of the function must be an Eigen matrix representing the Jacobian space");

    // TODO: Relax this requirement - functions should be able to go from one scalar type to another
    static_assert(std::is_same<typename utility::FunctionType<Function>::Output::Scalar,
                               typename utility::FunctionType<Function>::template Input<0>::Scalar>::value,
                  "Scalar type used int the input and output vectors must be the same");

    DifferentiableFunction(const Function& f, const Jacobian& j) noexcept : F(f), J(j) { }
    DifferentiableFunction(Function&& f, Jacobian&& j) noexcept : F(std::move(f)), J(std::move(j)) { }

    // Constructors that use the state spaces as "tags" for tagged dispatch
    // TODO: Can these be removed with some clever CTAD?
    DifferentiableFunction(const To& /*unused*/, const From& /*unused*/, const Function& f, const Jacobian& j) noexcept :
        F(f),
        J(j) { }
    DifferentiableFunction(const To& /*unused*/, const From& /*unused*/, Function&& f, Jacobian&& j) noexcept :
        F(std::move(f)),
        J(std::move(j)) { }

    DifferentiableFunction(const DifferentiableFunction& copy) noexcept = default;
    DifferentiableFunction(DifferentiableFunction&& move) noexcept = default;

    ~DifferentiableFunction() noexcept = default;

    DifferentiableFunction& operator=(const DifferentiableFunction& copy) noexcept = default;
    DifferentiableFunction& operator=(DifferentiableFunction&& move) noexcept = default;

    template <typename... AdditionalParameters>
    Output operator()(const Input& input, const AdditionalParameters&... augmented_state) const noexcept {
        return F(input, augmented_state...);
    }

    template <typename... AdditionalParameters>
    JacobianOutput jacobian(const Input& input, const AdditionalParameters&... augmented_state) const noexcept {
        return J(input, augmented_state...);
    }

    template <typename... AdditionalParameters>
    Mean<To, Scalar> operator()(const Mean<From, Scalar>& input,
                                const AdditionalParameters&... augmented_state) const noexcept {
        return Mean<To, Scalar>(F(input.mean, augmented_state...));
    }

    template <typename... AdditionalParameters>
    MeanAndCovariance<To, Scalar> operator()(const MeanAndCovariance<From, Scalar>& input,
                                             const AdditionalParameters&... augmented_state) const noexcept {
        const auto jacobian_evaluated = J(input.mean, augmented_state...);
        return MeanAndCovariance<To, Scalar>(F(input.mean, augmented_state...),
                                             jacobian_evaluated * input.covariance * jacobian_evaluated.transpose());
    }

    template <int N_Points, typename... AdditionalParameters>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input,
                                                     const AdditionalParameters&... augmented_state) const noexcept {
        typename WeightedSamples<To, Scalar, N_Points>::SamplesMatrix transformed_samples;
        for (size_t i = 0; i < N_Points; i++) {
            transformed_samples.col(i) = F(input.samples.col(i), augmented_state...);
        }
        return WeightedSamples<To, Scalar, N_Points>(std::move(transformed_samples), input.weights);
    }

    Function F;
    Jacobian J;
};

} // namespace vague