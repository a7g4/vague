#pragma once

#include <array>

namespace vague::state_spaces {

struct CartesianPos2D {
    enum Elements { X, Y, N };
    constexpr static std::array<std::size_t, 0> ANGLES {};
};

struct CartesianPosYaw2D {
    enum Elements { X, Y, YAW, N };
    constexpr static std::array<std::size_t, 1> ANGLES {YAW};
};

struct Box2D {
    enum Elements { X, Y, L, W, YAW, N };
    constexpr static std::array<std::size_t, 1> ANGLES {YAW};
};

struct CartesianPosVel2D {
    enum Elements { X, Y, V_X, V_Y, N };
    constexpr static std::array<std::size_t, 0> ANGLES {};
};

struct RangeAzimuthRangeRate {
    enum Elements { RANGE, AZIMUTH, RANGE_RATE, N };
    constexpr static std::array<std::size_t, 1> ANGLES {AZIMUTH};
};

} // namespace vague::state_spaces