#include "search.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

#include "eval.h"
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
    if (stop_.load(std::memory_order_relaxed)) return true;
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

        if (stop_.load(std::memory_order_relaxed)) return 0;
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
            if (stop_.load(std::memory_order_relaxed)) return 0;
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

        if (stop_.load(std::memory_order_relaxed)) return 0;

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

chess::Move Searcher::think(Board board, const SearchLimits& limits) {
    limits_ = limits;
    stop_.store(false);
    nodes_ = 0;
    sel_depth_ = 0;
    start_ = Clock::now();
    side_ = static_cast<int>(board.sideToMove());
    TT.new_search();

    soft_time_ms_ = (limits.infinite || limits.depth || limits.nodes) ? 0 : compute_move_time(limits, side_);

    // Decay history between searches so it stays relevant.
    for (auto& a : history_)
        for (auto& b2 : a)
            for (auto& h : b2) h /= 2;
    std::fill(&killers_[0][0], &killers_[0][0] + sizeof(killers_) / sizeof(uint16_t), 0);

    // Fallback: first legal move, in case we get stopped before depth 1 finishes.
    Movelist root_moves;
    movegen::legalmoves(root_moves, board);
    Move best = root_moves.empty() ? Move() : root_moves[0];

    int max_depth = limits.depth ? std::min(limits.depth, MAX_PLY) : MAX_PLY;
    Value prev = VALUE_NONE;

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

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_).count();
        uint64_t nps = ms > 0 ? (nodes_ * 1000 / ms) : nodes_;

        std::string pv;
        for (int i = 0; i < pv_len_[0]; ++i) {
            if (i) pv += ' ';
            pv += uci::moveToUci(Move(pv_table_[0][i]));
        }
        std::cout << "info depth " << depth << " seldepth " << sel_depth_
                  << " score " << score_to_uci(score) << " nodes " << nodes_ << " nps " << nps
                  << " hashfull " << TT.hashfull() << " time " << ms;
        if (!pv.empty()) std::cout << " pv " << pv;
        std::cout << "\n" << std::flush;

        if (is_mate_score(score) && depth >= 2 * ((VALUE_MATE - std::abs(score) + 1) / 2)) break;
        if (aborted) break;
        // If we have spent most of our soft budget, don't start another iteration.
        if (soft_time_ms_ > 0 && ms * 100 >= soft_time_ms_ * 55) break;
    }

    return best;
}

}  // namespace eng
