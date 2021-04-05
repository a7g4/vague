#pragma once

#include <array>

namespace vague::state_spaces {

struct CartesianPos2D {
    enum Elements { X, Y, N };
    constexpr static std::array<size_t, 0> Angles {};
};

struct CartesianPosYaw2D {
    enum Elements { X, Y, Yaw, N };
    constexpr static std::array<size_t, 1> Angles { Yaw };
};

struct Box2D {
    enum Elements { X, Y, L, W, Yaw, N };
    constexpr static std::array<size_t, 1> Angles { Yaw };
};

struct CartesianPosVel2D {
    enum Elements {X, Y, V_X, V_Y, N };
    constexpr static std::array<size_t, 0> Angles {};
};

struct RangeAzimuthRangeRate {
    enum Elements {Range, Azimuth, RangeRate, N};
    constexpr static std::array<size_t, 1> Angles {Azimuth}; 
};

}