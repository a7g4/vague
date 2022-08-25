#include "vague/arbitrary_function.hpp"
#include "vague/differentiable_function.hpp"
#include "vague/estimate.hpp"
#include "vague/linear_function.hpp"
#include "vague/state_spaces.hpp"

#include <iostream>

struct From {
    enum Elements { E1, N };
    constexpr static std::array<size_t, 0> ANGLES {};
};

struct To {
    enum Elements { E1, E2, N };
    constexpr static std::array<size_t, 0> ANGLES {};
};

int main() {
    using Input = Eigen::Matrix<double, 1, 1>;
    using Output = Eigen::Matrix<double, 2, 1>;

    vague::LinearFunction<To, From, double> f(Eigen::Matrix<double, 2, 1> {{1}, {2}});

    vague::DifferentiableFunction<To, From, std::function<Output(const Input&)>, std::function<Output(const Input&)>> df(
        [](const Input& i) -> Output {
            return Output {i[0], i[0] * i[0]};
        },
        [](const Input& i) -> Output {
            return Output {1, 2 * i[0]};
        });

    vague::ArbitraryFunction<To, From, std::function<Output(const Input&)>> af([](const Input& i) -> Output {
        return Output {i[0], i[0] * i[0]};
    });

    std::cout << "Raw Eigen\n";
    Input i {1.5};
    std::cout << i << std::endl;
    std::cout << f(i).transpose() << std::endl;
    std::cout << df(i).transpose() << std::endl;
    std::cout << af(i).transpose() << std::endl;

    std::cout << "Mean\n";
    vague::Mean<From, double> m(i);
    std::cout << m.mean << std::endl;
    std::cout << f(m).mean.transpose() << std::endl;
    std::cout << df(m).mean.transpose() << std::endl;
    std::cout << af(m).mean.transpose() << std::endl;

    std::cout << "MeanAndCovariance\n";
    vague::MeanAndCovariance<From, double> mac(i, Eigen::Matrix<double, 1, 1>(1));
    std::cout << mac.mean << std::endl;
    std::cout << f(mac).mean.transpose() << std::endl;
    std::cout << df(mac).mean.transpose() << std::endl;
    std::cout << af(m).mean.transpose() << std::endl;

    // filter f<pos_vel_2d>
    // predicted = f.predict<r_az_rr>(radar x, y, yaw);
    // f.update(measurement, predicted);

    return 0;
}