#pragma once
#include "../third_party/chess-library/chess.hpp"
#include "types.h"

namespace eng {

// Static evaluation from the side-to-move's perspective (positive == good for
// the player to move). M1 uses a tapered PeSTO hand-crafted evaluation; once an
// NNUE network is loaded (M3) this dispatches to the network instead.
Value evaluate(const chess::Board& board);

// Hand-crafted evaluation, always available as a fallback / baseline.
Value evaluate_hce(const chess::Board& board);

}  // namespace eng
