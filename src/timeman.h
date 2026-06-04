#pragma once
#include <algorithm>
#include <cstdint>

namespace eng {

// Parameters parsed from a UCI "go" command.
struct SearchLimits {
    int64_t time[2] = {0, 0};   // [white, black] remaining ms
    int64_t inc[2]  = {0, 0};   // [white, black] increment ms
    int64_t movetime = 0;       // fixed ms for this move
    int     movestogo = 0;      // moves until next time control (0 == sudden death)
    int     depth = 0;          // fixed depth (0 == unlimited)
    uint64_t nodes = 0;         // node cap (0 == unlimited)
    bool    infinite = false;   // search until "stop"
    int     human_style = 0;    // 0 = pure strength; >0 = prefer human moves within a margin
};

// Compute a soft time budget (ms) for this move. side: 0 white, 1 black.
inline int64_t compute_move_time(const SearchLimits& lim, int side) {
    if (lim.movetime > 0) return std::max<int64_t>(1, lim.movetime - 20);

    int64_t remaining = lim.time[side];
    int64_t inc = lim.inc[side];
    if (remaining <= 0) return 0;   // no clock info: rely on depth/nodes/infinite

    int mtg = lim.movestogo > 0 ? lim.movestogo : 30;
    int64_t budget = remaining / mtg + inc * 3 / 4;

    // Never spend more than ~40% of the remaining time on one move; keep a buffer.
    budget = std::min(budget, remaining * 2 / 5);
    budget -= 20;   // communication / overhead margin
    return std::max<int64_t>(1, budget);
}

}  // namespace eng
