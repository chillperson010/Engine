#pragma once
#include <cstdint>

// Shared engine constants and types.
namespace eng {

using Value = int32_t;

constexpr Value VALUE_INF      = 32000;
constexpr Value VALUE_MATE     = 31000;
constexpr Value VALUE_MATE_MAX = VALUE_MATE - 256;  // mate scores live above this
constexpr Value VALUE_NONE     = 32001;
constexpr Value VALUE_DRAW     = 0;

constexpr int MAX_PLY = 246;

// Mate distance helpers: store mate-in-N as a score, distance from the root.
constexpr bool is_mate_score(Value v) {
    return v >= VALUE_MATE_MAX || v <= -VALUE_MATE_MAX;
}

}  // namespace eng
