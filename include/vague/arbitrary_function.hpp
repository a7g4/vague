#pragma once

#include <Eigen/Core>
#include "vague/utility.hpp"
#include "vague/state_spaces.hpp"
#include "vague/estimate.hpp"
#include <utility>

namespace vague {

template <typename ToT, typename FromT, typename FunctionT>
struct ArbitraryFunction {

    using To = ToT;
    using From = FromT;
    using Function = FunctionT;

    constexpr static size_t DIM_RANGE = To::N;
    constexpr static size_t DIM_DOMAIN = From::N;

    using Scalar = typename utility::FunctionType<Function>::Output::Scalar;
    using Input = Eigen::Matrix<Scalar, DIM_DOMAIN, 1>;
    using Output = Eigen::Matrix<Scalar, DIM_RANGE, 1>;

    static_assert(std::is_same<Input, typename utility::FunctionType<Function>::template Input<0>>::value,
                  "First input to the function must be an Eigen vector representing the 'From' space");

    static_assert(std::is_same<Output, typename utility::FunctionType<Function>::Output>::value,
                  "Output of the function must be an Eigen vector representing the 'To' space");

    // TODO: Relax this requirement - functions should be able to go from one scalar type to another
    static_assert(std::is_same<typename utility::FunctionType<Function>::Output::Scalar,
                               typename utility::FunctionType<Function>::template Input<0>::Scalar>::value,
                  "Scalar type used int the input and output vectors must be the same");

    ArbitraryFunction(const Function& f) noexcept : F(f) { }
    ArbitraryFunction(Function&& f) noexcept : F(std::move(f)) { }
    
    ArbitraryFunction(const ArbitraryFunction& copy) noexcept = default;
    ArbitraryFunction(ArbitraryFunction&& move) noexcept = default;

    // Constructors that use the state spaces as "tags" for tagged dispatch
    // TODO: Can these be removed with some clever CTAD?
    ArbitraryFunction(const To&, const From&, const Function& f) noexcept : F(f) { }
    ArbitraryFunction(const To&, const From&, Function&& f) noexcept : F(std::move(f)) { }

    template <typename ... AdditionalParameters>
    Output operator()(const Input& from, const AdditionalParameters&... additional_parameters) const noexcept {
        return F(from, additional_parameters...);
    }

    template <typename ... AdditionalParameters>
    Mean<To, Scalar> operator()(const Mean<From, Scalar>& from, const AdditionalParameters&... additional_parameters) const noexcept {
        return F(from.mean, additional_parameters...);
    }

    template <int N_Points, typename ... AdditionalParameters>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input, const AdditionalParameters&... additional_parameters) const noexcept {
        typename WeightedSamples<To, Scalar, N_Points>::SamplesMatrix transformed_points;
        for (size_t i = 0; i < N_Points; i++) {
            transformed_points.col(i) = F(input.samples.col(i), additional_parameters...);
        }
        return WeightedSamples<To, Scalar, N_Points>(std::move(transformed_points), input.weights);
    }

    Function F;
};

}