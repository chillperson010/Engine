#pragma once
#include <string>

#include "../third_party/chess-library/chess.hpp"

// Human move-prediction (policy) network: "learn like a human, play like one".
// A factored policy over the side-to-move's perspective producing a per-square
// from/to logit. Used to (a) order moves the way a titled human would consider
// them, and (b) bias root selection toward human moves (HumanStyle). It never
// feeds a value into the TT, so float math is fine.
namespace eng::policy {

bool is_loaded();
bool load(const std::string& path);

// Per-position from/to logits (computed once per node).
struct Eval {
    float from_logit[64];
    float to_logit[64];
};

void compute(const chess::Board& board, Eval& out);

// Human-likeness score of a move given precomputed per-position logits.
float move_score(const chess::Board& board, const Eval& e, chess::Move m);

}  // namespace eng::policy
