#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.h"

namespace eng {

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    uint16_t key16 = 0;   // upper bits of the Zobrist key for collision checks
    uint16_t move  = 0;   // best move (chess::Move raw 16-bit move payload)
    int16_t  value = 0;   // stored score
    int16_t  eval  = 0;   // static eval (reserved for later use)
    uint8_t  depth = 0;
    uint8_t  bound = BOUND_NONE;
    uint8_t  gen   = 0;   // generation for aging
};

// A simple fixed-bucket transposition table with depth-preferred replacement.
class TranspositionTable {
public:
    void resize(size_t mb);   // (re)allocate to the requested size in MiB
    void clear();
    void new_search() { gen_ = (gen_ + 1) & 0xFF; }

    // Probe; sets `hit` and returns the entry slot (always valid to read).
    TTEntry* probe(uint64_t key, bool& hit);

    void store(uint64_t key, uint16_t move, Value value, Value eval, int depth, Bound bound, int ply);

    int hashfull() const;     // permille of entries used by the current search

private:
    std::vector<TTEntry> table_;
    size_t mask_ = 0;
    uint8_t gen_ = 0;
};

extern TranspositionTable TT;

// Adjust mate scores when storing into / reading out of the TT so that
// "mate in N from here" stays correct regardless of search ply.
Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply);

}  // namespace eng
