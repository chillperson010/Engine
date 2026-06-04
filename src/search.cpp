#include "search.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "eval.h"
#include "policy.h"
#include "see.h"
#include "tt.h"

using namespace chess;

namespace eng {

Searcher g_searcher;

// ---- small helpers --------------------------------------------------------

static inline bool is_capture(const Board& b, Move m) {
    return m.typeOf() == Move::ENPASSANT ||
           (m.typeOf() != Move::CASTLING && b.at(m.to()) != Piece::NONE);
}

// Piece values for MVV-LVA / move ordering (PAWN..KING).
static const int PIECE_VAL[7] = {100, 320, 330, 500, 950, 20000, 0};

static inline int piece_val(const Board& b, Square sq) {
    Piece p = b.at(sq);
    if (p == Piece::NONE) return 0;
    return PIECE_VAL[static_cast<int>(p.type())];
}

static inline bool has_non_pawn_material(const Board& b, Color c) {
    return b.pieces(PieceType::KNIGHT, c).count() || b.pieces(PieceType::BISHOP, c).count() ||
           b.pieces(PieceType::ROOK, c).count() || b.pieces(PieceType::QUEEN, c).count();
}

void Searcher::resize_tt(size_t mb) { TT.resize(mb); }

void Searcher::clear() {
    TT.clear();
    std::fill(&killers_[0][0], &killers_[0][0] + sizeof(killers_) / sizeof(uint16_t), 0);
    std::fill(&history_[0][0][0], &history_[0][0][0] + sizeof(history_) / sizeof(int), 0);
}

bool Searcher::time_up() {
    if (stop_->load(std::memory_order_relaxed)) return true;
    if (limits_.nodes && nodes_ >= limits_.nodes) return true;
    if (soft_time_ms_ > 0) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
        if (ms >= soft_time_ms_) return true;
    }
    return false;
}

// ---- move ordering --------------------------------------------------------

void Searcher::score_moves(const Board& b, Movelist& moves, uint16_t tt_move, int ply) {
    int stm = static_cast<int>(b.sideToMove());
    for (auto& m : moves) {
        int s;
        if (m.move() == tt_move) {
            s = 30000;
        } else if (is_capture(b, m)) {
            int victim = (m.typeOf() == Move::ENPASSANT) ? 100 : piece_val(b, m.to());
            int attacker = piece_val(b, m.from());
            int mvv = victim * 8 - attacker;            // MVV-LVA
            // Winning/equal captures on top; losing captures (SEE<0) below quiets.
            s = see_ge(b, m, 0) ? (16000 + mvv) : (-12000 + mvv);
            if (m.typeOf() == Move::PROMOTION) s += 2000;
        } else if (m.typeOf() == Move::PROMOTION) {
            s = 15000 + static_cast<int>(m.promotionType());
        } else if (m.move() == killers_[ply][0]) {
            s = 8000;
        } else if (m.move() == killers_[ply][1]) {
            s = 7000;
        } else {
            int h = history_[stm][m.from().index()][m.to().index()];
            s = std::clamp(h, -6000, 6000);
        }
        m.setScore(static_cast<int16_t>(s));
    }
    std::sort(moves.begin(), moves.end(),
              [](const Move& a, const Move& b) { return a.score() > b.score(); });
}

// ---- quiescence -----------------------------------------------------------

Value Searcher::qsearch(Board& b, int ply, Value alpha, Value beta) {
    if ((++nodes_ & 2047) == 0 && time_up()) return 0;
    if (ply > sel_depth_) sel_depth_ = ply;
    if (ply >= MAX_PLY) return evaluate(b);

    Value stand = evaluate(b);
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;

    Movelist caps;
    movegen::legalmoves<movegen::MoveGenType::CAPTURE>(caps, b);
    score_moves(b, caps, 0, ply);

    Value best = stand;
    for (const auto& m : caps) {
        // Captures are SEE-ordered; once we reach losing captures (negative
        // score) the rest are losing too, so stop searching them in quiescence.
        if (m.score() < 0) break;

        // Delta pruning: skip captures that cannot raise alpha even optimistically.
        int gain = (m.typeOf() == Move::ENPASSANT) ? 100 : piece_val(b, m.to());
        if (stand + gain + 200 < alpha && m.typeOf() != Move::PROMOTION) continue;

        b.makeMove(m);
        Value score = -qsearch(b, ply + 1, -beta, -alpha);
        b.unmakeMove(m);

        if (stop_->load(std::memory_order_relaxed)) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) alpha = score;
            if (alpha >= beta) break;
        }
    }
    return best;
}

// ---- main search ----------------------------------------------------------

Value Searcher::search(Board& b, int depth, int ply, Value alpha, Value beta, bool cut_node) {
    const bool root = (ply == 0);
    const bool pv_node = (beta - alpha > 1);
    pv_len_[ply] = 0;

    if ((++nodes_ & 2047) == 0 && time_up()) return 0;
    if (ply > sel_depth_) sel_depth_ = ply;

    if (!root) {
        // Draw detection.
        if (b.isRepetition(1) || b.isHalfMoveDraw() || b.isInsufficientMaterial())
            return VALUE_DRAW;
        if (ply >= MAX_PLY) return evaluate(b);

        // Mate-distance pruning.
        alpha = std::max<Value>(alpha, -VALUE_MATE + ply);
        beta  = std::min<Value>(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    if (depth <= 0) return qsearch(b, ply, alpha, beta);

    // Transposition table probe.
    bool tt_hit = false;
    TTEntry* tte = TT.probe(b.hash(), tt_hit);
    uint16_t tt_move = tt_hit ? tte->move : 0;
    Value tt_value = tt_hit ? value_from_tt(tte->value, ply) : VALUE_NONE;

    if (!pv_node && tt_hit && tte->depth >= depth && tt_value != VALUE_NONE) {
        if (tte->bound == BOUND_EXACT) return tt_value;
        if (tte->bound == BOUND_LOWER && tt_value >= beta) return tt_value;
        if (tte->bound == BOUND_UPPER && tt_value <= alpha) return tt_value;
    }

    const bool in_check = b.inCheck();
    Value eval = in_check ? VALUE_NONE : evaluate(b);

    // Forward pruning (only in non-PV, non-check nodes).
    if (!pv_node && !in_check) {
        // Reverse futility / static null-move pruning.
        if (depth <= 6 && eval - 80 * depth >= beta && !is_mate_score(beta))
            return eval;

        // Null-move pruning.
        if (depth >= 3 && eval >= beta && has_non_pawn_material(b, b.sideToMove())) {
            int R = 3 + depth / 4;
            b.makeNullMove();
            Value null_score = -search(b, depth - R, ply + 1, -beta, -beta + 1, !cut_node);
            b.unmakeNullMove();
            if (stop_->load(std::memory_order_relaxed)) return 0;
            if (null_score >= beta) return is_mate_score(null_score) ? beta : null_score;
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, b);

    // Checkmate / stalemate.
    if (moves.empty()) return in_check ? (-VALUE_MATE + ply) : VALUE_DRAW;

    score_moves(b, moves, tt_move, ply);

    Value best = -VALUE_INF;
    uint16_t best_move = 0;
    Bound bound = BOUND_UPPER;
    int move_count = 0;
    int stm = static_cast<int>(b.sideToMove());

    for (const auto& m : moves) {
        ++move_count;
        const bool capture = is_capture(b, m);
        const bool quiet = !capture && m.typeOf() != Move::PROMOTION;

        // Late move pruning: at low depth, skip late quiet moves.
        if (!pv_node && !in_check && quiet && depth <= 4 && move_count > 4 + depth * depth &&
            !is_mate_score(best))
            continue;

        // Futility pruning of quiet moves near the horizon.
        if (!pv_node && !in_check && quiet && depth <= 4 && move_count > 1 &&
            eval != VALUE_NONE && eval + 100 + 90 * depth <= alpha && !is_mate_score(best))
            continue;

        b.makeMove(m);
        bool gives_check = b.inCheck();
        int ext = gives_check ? 1 : 0;          // check extension
        int new_depth = depth - 1 + ext;

        Value score;
        if (move_count == 1) {
            score = -search(b, new_depth, ply + 1, -beta, -alpha, false);
        } else {
            // Late move reductions for quiet, late moves.
            int R = 0;
            if (depth >= 3 && quiet && !gives_check) {
                R = 1 + (move_count > 6 ? 1 : 0) + (depth > 6 ? 1 : 0);
                if (pv_node) R = std::max(0, R - 1);
            }
            score = -search(b, new_depth - R, ply + 1, -alpha - 1, -alpha, true);
            if (score > alpha && R > 0)
                score = -search(b, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);
            if (score > alpha && score < beta)
                score = -search(b, new_depth, ply + 1, -beta, -alpha, false);
        }
        b.unmakeMove(m);

        if (stop_->load(std::memory_order_relaxed)) return 0;

        if (score > best) {
            best = score;
            best_move = m.move();
            if (pv_node) {
                pv_table_[ply][0] = m.move();
                int clen = pv_len_[ply + 1];
                for (int i = 0; i < clen; ++i) pv_table_[ply][i + 1] = pv_table_[ply + 1][i];
                pv_len_[ply] = clen + 1;
            }
            if (score > alpha) {
                alpha = score;
                bound = BOUND_EXACT;
                if (alpha >= beta) {
                    bound = BOUND_LOWER;
                    // Update killers / history for quiet cutoffs.
                    if (quiet) {
                        if (killers_[ply][0] != m.move()) {
                            killers_[ply][1] = killers_[ply][0];
                            killers_[ply][0] = m.move();
                        }
                        history_[stm][m.from().index()][m.to().index()] += depth * depth;
                    }
                    break;
                }
            }
        }
    }

    TT.store(b.hash(), best_move, best, eval, depth, bound, ply);
    return best;
}

// ---- iterative deepening / UCI driver -------------------------------------

static std::string score_to_uci(Value v) {
    if (is_mate_score(v)) {
        int mate_in = (v > 0) ? (VALUE_MATE - v + 1) / 2 : -(VALUE_MATE + v + 1) / 2;
        return "mate " + std::to_string(mate_in);
    }
    return "cp " + std::to_string(v);
}

// Explicit root search used only for the human-style pass: scores every root
// move (exact_all -> full window, no sibling cutoffs) so we can compare them.
// Relies on a warm TT (filled by the main search) to stay cheap.
Value Searcher::root_search(Board& b, int depth, Value alpha, Value beta, bool exact_all) {
    Value best = -VALUE_INF;
    pv_len_[0] = 0;
    int i = 0;
    for (auto& m : root_moves_) {
        b.makeMove(m);
        Value score;
        if (i == 0 || exact_all) {
            score = -search(b, depth - 1, 1, -beta, -alpha, false);
        } else {
            score = -search(b, depth - 1, 1, -alpha - 1, -alpha, true);
            if (score > alpha && score < beta)
                score = -search(b, depth - 1, 1, -beta, -alpha, false);
        }
        b.unmakeMove(m);
        root_scores_[i] = score;
        if (score > best) {
            best = score;
            pv_table_[0][0] = m.move();
            int clen = pv_len_[1];
            for (int k = 0; k < clen; ++k) pv_table_[0][k + 1] = pv_table_[1][k];
            pv_len_[0] = clen + 1;
        }
        if (!exact_all && score > alpha) alpha = score;
        ++i;
        if (i >= 256 || stop_->load(std::memory_order_relaxed)) break;
    }
    return best;
}

chess::Move Searcher::think(Board board, const SearchLimits& limits) {
    own_stop_.store(false);
    stop_ = &own_stop_;
    report_ = true;
    TT.new_search();
    return run(std::move(board), limits);
}

chess::Move Searcher::run(Board board, const SearchLimits& limits) {
    limits_ = limits;
    nodes_ = 0;
    sel_depth_ = 0;
    start_ = Clock::now();
    side_ = static_cast<int>(board.sideToMove());

    soft_time_ms_ = (limits.infinite || limits.depth || limits.nodes) ? 0 : compute_move_time(limits, side_);

    // Decay history between searches so it stays relevant.
    for (auto& a : history_)
        for (auto& b2 : a)
            for (auto& h : b2) h /= 2;
    std::fill(&killers_[0][0], &killers_[0][0] + sizeof(killers_) / sizeof(uint16_t), 0);

    // Fallback: first legal move, in case we get stopped before depth 1 finishes.
    movegen::legalmoves(root_moves_, board);
    Move best = root_moves_.empty() ? Move() : root_moves_[0];

    int max_depth = limits.depth ? std::min(limits.depth, MAX_PLY) : MAX_PLY;
    Value prev = VALUE_NONE;
    int completed_depth = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        Value alpha = -VALUE_INF, beta = VALUE_INF;
        Value score;

        if (depth >= 4 && !is_mate_score(prev) && prev != VALUE_NONE) {
            // Aspiration windows.
            int window = 25;
            while (true) {
                alpha = prev - window;
                beta = prev + window;
                score = search(board, depth, 0, alpha, beta, false);
                if (time_up()) break;
                if (score <= alpha) {
                    window *= 2;
                } else if (score >= beta) {
                    window *= 2;
                } else {
                    break;
                }
                if (window > 1000) {
                    score = search(board, depth, 0, -VALUE_INF, VALUE_INF, false);
                    break;
                }
            }
        } else {
            score = search(board, depth, 0, alpha, beta, false);
        }

        bool aborted = time_up();
        if (aborted && depth > 1) break;   // discard incomplete iteration

        if (pv_len_[0] > 0) best = Move(pv_table_[0][0]);
        prev = score;
        completed_depth = depth;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
        uint64_t nps = ms > 0 ? (nodes_ * 1000 / ms) : nodes_;

        std::string pv;
        for (int i = 0; i < pv_len_[0]; ++i) {
            if (i) pv += ' ';
            pv += uci::moveToUci(Move(pv_table_[0][i]));
        }
        if (report_) {
            std::cout << "info depth " << depth << " seldepth " << sel_depth_
                      << " score " << score_to_uci(score) << " nodes " << nodes_ << " nps " << nps
                      << " hashfull " << TT.hashfull() << " time " << ms;
            if (!pv.empty()) std::cout << " pv " << pv;
            std::cout << "\n" << std::flush;
        }

        if (is_mate_score(score) && depth >= 2 * ((VALUE_MATE - std::abs(score) + 1) / 2)) break;
        if (aborted) break;
        // If we have spent most of our soft budget, don't start another iteration.
        if (soft_time_ms_ > 0 && ms * 100 >= soft_time_ms_ * 55) break;
    }

    // Human-style pass: among root moves within a centipawn margin of the best,
    // play the one a titled human would most likely choose. Uses a cheap pass
    // over the (warm) TT with time/stop disabled so scores are reliable.
    if (limits_.human_style > 0 && policy::is_loaded() && root_moves_.size() > 1 &&
        completed_depth >= 1) {
        soft_time_ms_ = 0;
        limits_.nodes = 0;
        std::atomic<bool> nostop{false};
        std::atomic<bool>* saved = stop_;
        stop_ = &nostop;

        int hd = std::max(2, std::min(completed_depth, 10));
        root_search(board, hd, -VALUE_INF, VALUE_INF, true);

        Value bestS = -VALUE_INF;
        int nmoves = static_cast<int>(root_moves_.size());
        for (int i = 0; i < nmoves; ++i) bestS = std::max(bestS, root_scores_[i]);

        int margin = limits_.human_style * 2;   // cp tolerance traded for human style
        policy::Eval pe;
        policy::compute(board, pe);
        float best_human = -1e30f;
        Move chosen = best;
        for (int i = 0; i < nmoves; ++i) {
            if (root_scores_[i] >= bestS - margin && !is_mate_score(bestS)) {
                float h = policy::move_score(board, pe, root_moves_[i]);
                if (h > best_human) { best_human = h; chosen = root_moves_[i]; }
            }
        }
        best = chosen;
        stop_ = saved;
        if (report_)
            std::cout << "info string human-style pick " << uci::moveToUci(best) << "\n" << std::flush;
    }

    return best;
}

// ---- Lazy SMP coordinator -------------------------------------------------

static std::atomic<bool> g_shared_stop{false};

void request_stop() { g_shared_stop.store(true, std::memory_order_relaxed); }

// Cleared by the UCI thread before launching a search, so a "stop" arriving
// right after "go" can't be clobbered by the coordinator resetting it.
void clear_stop() { g_shared_stop.store(false, std::memory_order_relaxed); }

chess::Move search_best(const Board& board, const SearchLimits& limits, int threads) {
    if (threads < 1) threads = 1;
    TT.new_search();

    std::vector<std::unique_ptr<Searcher>> pool;
    for (int i = 0; i < threads; ++i) {
        auto s = std::make_unique<Searcher>();
        s->set_shared_stop(&g_shared_stop);
        s->set_report(i == 0);
        pool.push_back(std::move(s));
    }

    // Helper threads deepen "infinitely" until the shared stop is set; only the
    // main thread manages time and prints info. All share the global TT.
    SearchLimits helper = limits;
    helper.infinite = true;
    helper.movetime = 0;
    helper.depth = 0;
    helper.nodes = 0;

    std::vector<std::thread> ts;
    for (int i = 1; i < threads; ++i) {
        Searcher* w = pool[i].get();
        ts.emplace_back([w, board, helper]() mutable { w->run(board, helper); });
    }

    Move best = pool[0]->run(board, limits);          // main: time-managed + reporting
    g_shared_stop.store(true, std::memory_order_relaxed);
    for (auto& t : ts) t.join();
    return best;
}

}  // namespace eng
