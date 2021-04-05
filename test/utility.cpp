#include "catch.hpp"
#include "vague/utility.hpp"

int test(double a, float b, int& c);

TEST_CASE( "Function Input/Outputs types are correct", "[FunctionType]" ) {
    auto example = [](double, float, int&) -> int { return 0; };

    REQUIRE( std::is_same<int, vague::utility::FunctionType<decltype(example)>::Output>::value );

    REQUIRE( std::is_same<double, vague::utility::FunctionType<decltype(example)>::Input<0>>::value );
    REQUIRE( std::is_same<float, vague::utility::FunctionType<decltype(example)>::Input<1>>::value );
    REQUIRE( std::is_same<int, vague::utility::FunctionType<decltype(example)>::Input<2>>::value );
}