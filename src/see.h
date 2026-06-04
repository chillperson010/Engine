#pragma once
#include "../third_party/chess-library/chess.hpp"

namespace eng {

// Static Exchange Evaluation: the material the side to move gains (in
// centipawns) from the capture sequence on `move`'s target square, assuming
// both sides recapture with their least valuable piece. Used for move ordering
// and for pruning losing captures in quiescence.
int see(const chess::Board& board, chess::Move move);

inline bool see_ge(const chess::Board& board, chess::Move move, int threshold) {
    return see(board, move) >= threshold;
}

}  // namespace eng
