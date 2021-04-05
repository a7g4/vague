cmake_minimum_required(VERSION 3.21)

project(vague LANGUAGES CXX VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

find_package(Eigen3 3.4 REQUIRED)
find_package(Catch2 REQUIRED)

add_library(vague INTERFACE)
add_dependencies(vague INTERFACE Eigen3)
target_include_directories(vague INTERFACE include/)

include(CTest)
include(Catch)


# TODO: Why aren't the libraries and include directories not pulled in from the target vague
add_executable(example_bin test/example_bin.cpp)
target_link_libraries(example_bin PUBLIC vague)
target_include_directories(example_bin PUBLIC ${EIGEN3_INCLUDE_DIR})

add_executable(benchmark_stuff test/benchmark.cpp)
target_link_libraries(benchmark_stuff PUBLIC vague)
target_link_libraries(benchmark_stuff PRIVATE Catch2::Catch2WithMain)
# catch_discover_tests(test_benchmark)

add_executable(test_sample test/sample.cpp)
target_link_libraries(test_sample PUBLIC vague)
target_link_libraries(test_sample PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_sample)

function(setup_test test_name)
    add_executable(${test_name} test/${test_name}.cpp)
    target_link_libraries(${test_name} PUBLIC vague)
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
    catch_discover_tests(${test_name})
endfunction()

setup_test(test_arbitrary_function)

add_executable(test_linear_function test/test_linear_function.cpp)
target_link_libraries(test_linear_function PUBLIC vague)
target_link_libraries(test_linear_function PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_linear_function)

add_executable(test_differentiable_function test/test_differentiable_function.cpp)
target_link_libraries(test_differentiable_function PUBLIC vague)
target_link_libraries(test_differentiable_function PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_differentiable_function)

add_executable(test_utility test/utility.cpp)
target_link_libraries(test_utility PUBLIC vague)
target_link_libraries(test_utility PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_utility)

add_executable(test_angle_statistics test/test_angle_statistics.cpp)
target_link_libraries(test_angle_statistics PUBLIC vague)
target_link_libraries(test_angle_statistics PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_angle_statistics)

add_executable(test_state_estimator test/test_state_estimator.cpp)
target_link_libraries(test_state_estimator PUBLIC vague)
target_link_libraries(test_state_estimator PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_state_estimator)