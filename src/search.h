#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "../third_party/chess-library/chess.hpp"
#include "timeman.h"
#include "types.h"

namespace eng {

class Searcher {
public:
    Searcher() = default;

    void resize_tt(size_t mb);
    void clear();   // clear TT + history between games (ucinewgame)

    // Single-threaded search: resets state, runs iterative deepening, returns
    // the best move (prints UCI "info"). Used by bench and the 1-thread path.
    chess::Move think(chess::Board board, const SearchLimits& limits);

    // One worker's iterative-deepening loop. Does NOT reset the shared stop flag
    // or the TT generation (the coordinator does). `report_` controls printing.
    chess::Move run(chess::Board board, const SearchLimits& limits);

    void stop() { stop_->store(true, std::memory_order_relaxed); }
    void set_shared_stop(std::atomic<bool>* p) { stop_ = p; }
    void set_report(bool r) { report_ = r; }

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
    std::atomic<bool> own_stop_{false};
    std::atomic<bool>* stop_ = &own_stop_;   // points to a shared flag under SMP
    bool report_ = true;
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

// Lazy SMP coordinator: run `threads` workers (main + helpers) sharing the
// global TT and a single stop flag; the main worker manages time and reporting.
// Returns the best move. threads<=1 is the plain single-threaded search.
chess::Move search_best(const chess::Board& board, const SearchLimits& limits, int threads);

// Signal a running search_best() to stop (used by the UCI "stop" command).
void request_stop();

// Clear the stop flag; call before launching a search (see search.cpp).
void clear_stop();

}  // namespace eng
