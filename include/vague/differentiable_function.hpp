#pragma once

#include <Eigen/Core>
#include "vague/utility.hpp"

namespace vague {

template <typename FromT, typename ToT, typename FunctionT, typename JacobianT>
struct DifferentiableFunction {
    
    using From = FromT;
    using To = ToT;
    using Function = FunctionT;
    using Jacobian = JacobianT;

    constexpr static size_t DIM_RANGE = To::N;
    constexpr static size_t DIM_DOMAIN = From::N;

    using Scalar = typename utility::FunctionType<Function>::Output::Scalar;
    using Input = Eigen::Matrix<Scalar, DIM_DOMAIN, 1>;
    using Output = Eigen::Matrix<Scalar, DIM_RANGE, 1>;
    using JacobianOutput = Eigen::Matrix<Scalar, DIM_RANGE, DIM_DOMAIN>;

    static_assert(std::is_same<Input,
                               typename utility::FunctionType<Function>::template Input<0>>::value);
    static_assert(std::is_same<Input,
                               typename utility::FunctionType<Jacobian>::template Input<0>>::value);

    static_assert(std::is_same<Output,
                               typename utility::FunctionType<Function>::Output>::value);
    static_assert(std::is_same<JacobianOutput,
                               typename utility::FunctionType<Jacobian>::Output>::value);

    static_assert(std::is_same<typename utility::FunctionType<Function>::Output::Scalar,
                               typename utility::FunctionType<Function>::template Input<0>::Scalar>::value);

    DifferentiableFunction(const Function& f, const Jacobian& j) : F(f), J(j) { }
    DifferentiableFunction(Function&& f, Jacobian&& j) : F(std::move(f)), J(std::move(j)) { }

    // Constructors that use the state spaces as "tags" for tagged dispatch
    // TODO: Can these be removed with some clever CTAD?
    DifferentiableFunction(const From&, const To&, const Function& f, const Jacobian& j) : F(f), J(j) { }
    DifferentiableFunction(const From&, const To&, Function&& f, Jacobian&& j) : F(std::move(f)), J(std::move(j)) { }

    template <typename ... AugmentedState>
    Output operator()(const Input& input, const AugmentedState&... augmented_state) const {
        return F(input, augmented_state...);
    }

    template <typename ... AugmentedState>
    JacobianOutput jacobian(const Input& input, const AugmentedState&... augmented_state) const {
        return J(input, augmented_state...);
    }

    template <typename ... AugmentedState>
    Mean<To, Scalar> operator()(const Mean<From, Scalar>& input, const AugmentedState&... augmented_state) const {
        return Mean<To, Scalar>(F(input.mean, augmented_state...));
    }

    template <typename ... AugmentedState>
    MeanAndCovariance<To, Scalar> operator()(const MeanAndCovariance<From, Scalar>& input, const AugmentedState&... augmented_state) const {
        const auto jacobian_evaluated = J(input.mean, augmented_state...);
        return MeanAndCovariance<To, Scalar>(F(input.mean, augmented_state...),
                                             jacobian_evaluated * input.covariance * jacobian_evaluated.transpose());
    }

    template <int N_Points, typename ... AugmentedState>
    WeightedSamples<To, Scalar, N_Points> operator()(const WeightedSamples<From, Scalar, N_Points>& input, const AugmentedState&... augmented_state) const {
        typename WeightedSamples<To, Scalar, N_Points>::SamplesMatrix transformed_samples;
        for (size_t i = 0; i < N_Points; i++) {
            transformed_samples.col(i) = F(input.samples.col(i), augmented_state...);
        }
        return WeightedSamples<To, Scalar, N_Points>(std::move(transformed_samples), input.weights);
    }

    Function F;
    Jacobian J;
};

// Deduction rule for when the template arguments looked like:
//      template <typename Scalar, size_t DIM_RANGE, size_t DIM_DOMAIN, typename Function, typename Jacobian>
// template <typename Function, typename Jacobian>
// DifferentiableFunction(Function, Jacobian) ->
//     DifferentiableFunction<double,
//                            size_t(utility::FunctionType<Function>::Output::RowsAtCompileTime),
//                            size_t(utility::FunctionType<Function>::template Input<0>::RowsAtCompileTime),
//                            Function,
//                            Jacobian>;
}