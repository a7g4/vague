#pragma once

#include <Eigen/Core>
#include "vague/utility.hpp"
#include "vague/state_spaces.hpp"
#include "vague/estimate.hpp"

namespace vague {

template <typename FromT, typename ToT, typename FunctionT>
struct ArbitraryFunction {

    using From = FromT;
    using To = ToT;
    using Function = FunctionT;

    constexpr static size_t DIM_RANGE = To::N;
    constexpr static size_t DIM_DOMAIN = From::N;

    using Scalar = typename utility::FunctionType<Function>::Output::Scalar;
    using Input = Eigen::Matrix<Scalar, DIM_DOMAIN, 1>;
    using Output = Eigen::Matrix<Scalar, DIM_RANGE, 1>;

    static_assert(std::is_same<Input,
                               typename utility::FunctionType<Function>::template Input<0>>::value);

    static_assert(std::is_same<Output,
                               typename utility::FunctionType<Function>::Output>::value);

    static_assert(std::is_same<typename utility::FunctionType<Function>::Output::Scalar,
                               typename utility::FunctionType<Function>::template Input<0>::Scalar>::value);

    ArbitraryFunction(const Function& f) : F(f) { }
    ArbitraryFunction(Function&& f) : F(std::move(f)) { }
    
    ArbitraryFunction(const ArbitraryFunction& copy) = default;
    ArbitraryFunction(ArbitraryFunction&& move) = default;

    // Constructors that use the state spaces as "tags" for tagged dispatch
    // TODO: Can these be removed with some clever CTAD?
    ArbitraryFunction(const From&, const To&, const Function& f) : F(f) { }
    ArbitraryFunction(const From&, const To&, Function&& f) : F(std::move(f)) { }

    template <typename ... AugmentedState>
    Output operator()(const Input& from, const AugmentedState&... augmented_state) const {
        return F(from, augmented_state...);
    }

    template <typename ... AugmentedState>
    Mean<To, Scalar> operator()(const Mean<From, Scalar>& from, const AugmentedState&... augmented_state) const {
        return F(from.mean, augmented_state...);
    }

    template <int N_Points, typename ... AugmentedState>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input, const AugmentedState&... augmented_state) const {
        typename WeightedSamples<To, Scalar, N_Points>::SamplesMatrix transformed_points;
        for (size_t i = 0; i < N_Points; i++) {
            transformed_points.col(i) = F(input.samples.col(i), augmented_state...);
        }
        return WeightedSamples<To, Scalar, N_Points>(std::move(transformed_points), input.weights);
    }

    Function F;
};

}