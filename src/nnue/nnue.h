#pragma once
#include <string>

#include "../../third_party/chess-library/chess.hpp"
#include "../types.h"

// NNUE evaluation. M1 ships a stub (never loaded) so the engine runs on the
// hand-crafted evaluation; M3 fills in the HalfKP accumulator + quantized SIMD
// forward pass and a .nnue loader. The eval/search code only touches this
// interface, so dropping in the real implementation needs no call-site changes.
namespace eng::nnue {

// Returns true once a network has been successfully loaded.
bool is_loaded();

// Load a network from a .nnue file. Returns false on any failure (and leaves
// the engine on the hand-crafted evaluation). Stub always returns false in M1.
bool load(const std::string& path);

// Evaluate from the side-to-move's perspective. Only valid when is_loaded().
Value evaluate(const chess::Board& board);

}  // namespace eng::nnue
