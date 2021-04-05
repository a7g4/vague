#pragma once

#include <type_traits>
#include <tuple>
#include <cstddef>

namespace vague::utility {

template<typename T>
struct remove_cvref_t {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
struct FunctionType : public FunctionType<decltype(&T::operator())>
{};

template <typename T, typename ReturnType, typename... Args>
struct FunctionType<ReturnType(T::*)(Args...) const>
{
    constexpr static size_t Inputs = sizeof...(Args);

    template <size_t i>
    using Input = typename remove_cvref_t<typename std::tuple_element<i, std::tuple<Args...,void>>::type>::type;

    using Output = typename remove_cvref_t<ReturnType>::type;
};

template <typename T, typename... Args>
struct FunctionTypeCallableWith
{
private:
    template <typename C,
              typename = decltype( std::declval<C>().operator()(std::declval<Args>()...) )>
    static std::true_type test(int);
    template <typename C>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

}