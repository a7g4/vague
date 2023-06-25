#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace vague::utility {

template <typename T>
struct RemoveCvrefT {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
struct FunctionType : public FunctionType<decltype(&T::operator())> { };

template <typename T, typename ReturnType, typename... Args>
struct FunctionType<ReturnType (T::*)(Args...) const> {
    constexpr static size_t INPUTS = sizeof...(Args);

    template <size_t i>
    using Input = typename RemoveCvrefT<typename std::tuple_element<i, std::tuple<Args..., void>>::type>::type;

    using Output = typename RemoveCvrefT<ReturnType>::type;
};

template <typename T, typename... Args>
struct FunctionTypeCallableWith {
private:
    template <typename C, typename = decltype(std::declval<C>().operator()(std::declval<Args>()...))>
    static std::true_type test(int);
    template <typename C>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value; // NOLINT(readability-identifier-naming)
};

} // namespace vague::utility