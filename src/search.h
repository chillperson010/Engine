#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

#include "../third_party/chess-library/chess.hpp"
#include "timeman.h"
#include "types.h"

namespace eng {

class Searcher {
public:
    Searcher() { resize_tt(16); }

    void resize_tt(size_t mb);
    void clear();   // clear TT + history between games (ucinewgame)

    // Run an iterative-deepening search and return the best move. Prints UCI
    // "info" lines during the search. Honours `limits` (time/depth/nodes/...).
    chess::Move think(chess::Board board, const SearchLimits& limits);

    void stop() { stop_.store(true, std::memory_order_relaxed); }

    uint64_t nodes() const { return nodes_; }

private:
    using Clock = std::chrono::steady_clock;

    Value search(chess::Board& b, int depth, int ply, Value alpha, Value beta, bool cut_node);
    Value qsearch(chess::Board& b, int ply, Value alpha, Value beta);

    void score_moves(const chess::Board& b, chess::Movelist& moves, uint16_t tt_move, int ply);
    bool time_up();

    SearchLimits limits_{};
    int64_t soft_time_ms_ = 0;
    Clock::time_point start_{};
    std::atomic<bool> stop_{false};
    uint64_t nodes_ = 0;
    int side_ = 0;
    int sel_depth_ = 0;

    // Move-ordering heuristics.
    static constexpr int MAXP = MAX_PLY + 8;
    uint16_t killers_[MAXP][2] = {};
    int history_[2][64][64] = {};      // [stm][from][to]

    // Principal variation for the root.
    uint16_t pv_table_[MAXP][MAXP] = {};
    int pv_len_[MAXP] = {};
};

extern Searcher g_searcher;

}  // namespace eng
